"""Assemble all completed R4 evidence; never promote a method configuration."""
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R4_20261024'

def md(d):
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |',
        *['| '+' | '.join(map(str,r))+' |' for r in d.itertuples(index=False,name=None)]])

def nav_report():
    folder=OUT/'nav_selector';state=json.loads((folder/'DONE.json').read_text())
    assert state['status']=='complete' and state['exact_action_field_support_differences']==0
    d=pd.read_csv(folder/'summary.csv');effects=pd.read_csv(folder/'contrasts.csv');rows=[]
    for grid in ['7x7','3x3']:
        for method in ['original','cached']:
            row={'grid':grid,'implementation':method}
            for metric,label in [('mean_total_ms','mean ms [95% CI]'),('p95_total_ms','p95 ms [95% CI]'),('mean_candidate_object_TTC_evaluations','candidate-object TTC evaluations')]:
                r=d[(d.grid==grid)&(d.method==method)&(d.metric==metric)].iloc[0]
                row[label]=f'{r.estimate:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]'
            rows.append(row)
    table=pd.DataFrame(rows);ratios=effects[effects.metric=='mean_total_speedup_ratio'].copy()
    ratios['speedup [95% CI]']=ratios.apply(lambda r:f'{r.cached_minus_original:.2f}× [{r.ci_lo:.2f}, {r.ci_hi:.2f}]',axis=1)
    ratio=ratios[['grid','speedup [95% CI]']]
    lines=['# Exact bottleneck-selector cache: same-host CPU timing','',
        'A separate fixed extension motivated by source algebra, before this benchmark. K31 selector, unchanged 15 trajectories, scores, probe grids and tie ordering. Same frozen K34/K20 1,000 navtest tokens / 125 units, explicitly post-hoc runtime/parity only. No quality selection, detector rerun or planner replacement.',
        'For each fixed candidate, s(S)=min(b,min_i h(i)); hence s(S+probe)=min(s(S),h(probe)). Original code recalculates every existing object for every probe. Cache s(S) and H once, batch the probe-only columns, and reuse H for binding support. All per-object TTC arithmetic still calls the original Selector.h.',
        'One CPU thread on VUW1; five warm-up repeats, three timed repetitions of both implementations and both grids with alternating order. Scope includes scene construction, selector initialization, probe projection, field and support; excludes disk loading and image/detector inference. Every original-vs-cached action, field box and support box/order is identical in 6,000 comparisons. 7x7 also agrees with stored K31 first-round action/counts.','',md(table),'',md(ratio),'',
        'Paired unit bootstrap 2,000 seed0, repetitions nested within units; pooled p95 recomputed per draw. All effects: contrasts.csv; raw rows: frames.csv. Speedup ratios refer to this selector stage, not total perception latency or PDM-Closed. The 15 fixed candidate trajectories and h-independence permit this reduction; arbitrary planners need separate derivation.',
        'Sources and hashes: sources_before.json / sources_after.json; exact token manifest and parity.csv. Isolated implementations: experiments/optimization_r4_20261024/selector_cache.py and nav_selector.py. Core and K34 artifacts remain unchanged.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return table,ratio,state

def main():
    navtable,navratio,navstate=nav_report()
    full=pd.concat([pd.read_csv(OUT/s/'summary.csv').assign(suite=s) for s in ['dev','readout']],ignore_index=True)
    ci=pd.concat([pd.read_csv(OUT/s/'contrasts.csv').assign(suite=s) for s in ['dev','readout']],ignore_index=True)
    intervals=pd.concat([pd.read_csv(OUT/s/'intervals.csv').assign(suite=s) for s in ['dev','readout']],ignore_index=True)
    assert len(full)==58 and set(full.units)=={152,300}
    full.to_csv(OUT/'all_rows.csv',index=False);ci.to_csv(OUT/'all_contrasts.csv',index=False);intervals.to_csv(OUT/'all_intervals.csv',index=False)
    choices=[('dev','cheap'),('dev','v6-S'),('dev','round1 combined'),('dev','round1 combined / none'),('dev','round1 combined / half'),
        ('dev','critical extent retention'),('dev','containment merge'),('dev','union coverage'),('dev','semantic consistency'),('dev','one-round extent'),
        ('readout','objectwise median'),('readout','hard-brake corroboration'),('readout','framewise median')]
    key=pd.concat([full[(full.suite==s)&(full.pipeline==n)] for s,n in choices],ignore_index=True)
    key.to_csv(OUT/'key_results.csv',index=False)
    table=key[['dataset','pipeline','missed','episodes','false_per_1000','false_braking_frames_per_1000','insufficient_loss_per_frame','G','mean_network_px']].copy()
    table['missed/required']=table.missed.astype(str)+'/'+table.episodes.astype(str);table=table.drop(columns=['missed','episodes'])
    for c in table.columns[2:-1]:table[c]=table[c].map(lambda x:f'{x:.5f}')
    keyci=ci[(ci.comparator=='round1 combined')&ci.metric.isin(['all_missed_rate','false_per_1000','false_braking_frames_per_1000','insufficient_loss_per_frame','overbraking_cost_per_frame','G','mean_network_px'])]
    keyci=key[['suite','dataset','pipeline']].merge(keyci,on=['suite','dataset','pipeline']);keyci.to_csv(OUT/'key_contrasts.csv',index=False)
    effect=keyci.copy();effect['delta [95% CI]']=effect.apply(lambda r:f'{r.candidate_minus_baseline:+.5f} [{r.ci_lo:+.5f}, {r.ci_hi:+.5f}]',axis=1)
    effect=effect[['suite','dataset','pipeline','metric','delta [95% CI]']]
    timing={};timing_ci={};latency=[]
    for folder in ['jetson','jetson_fast']:
        timing[folder]=pd.read_csv(OUT/folder/'summary.csv');timing_ci[folder]=pd.read_csv(OUT/folder/'contrasts.csv')
        latency.append(timing[folder].assign(window=folder))
    lat=pd.concat(latency,ignore_index=True);lat.to_csv(OUT/'jetson_results.csv',index=False)
    lt=lat[['window','dataset','pipeline','mean_ms','p95_ms','p99_ms','above50_pct','above66_7_pct','above100_pct']].copy()
    for c in lt.columns[3:]:lt[c]=lt[c].map(lambda x:f'{x:.2f}')
    effects=[]
    for folder,d in timing_ci.items():
        q=d[(d.pipeline=='objectwise median skip')&d.comparator.isin(['round1 combined','objectwise median full'])&(d.metric=='mean_ms')]
        for r in q.itertuples():effects.append(dict(window=folder,dataset=r.dataset,comparator=r.comparator,**{'delta ms [95% CI]':f'{r.candidate_minus_baseline:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]'}))
    le=pd.DataFrame(effects);le.to_csv(OUT/'jetson_key_effects.csv',index=False)
    mechanics=pd.read_csv(OUT/'mechanisms/summary.csv')
    mechanics['estimate [95% CI]']=mechanics.apply(lambda r:f'{100*r.estimate:.2f}% [{100*r.ci_lo:.2f}, {100*r.ci_hi:.2f}]',axis=1)
    mech=mechanics[['dataset','metric','estimate [95% CI]','numerator','denominator']]
    attribution=pd.concat([pd.read_csv(OUT/s/'rescue_attribution_summary.csv').assign(suite=s) for s in ['dev','readout']],ignore_index=True)
    attribution.to_csv(OUT/'all_episode_attribution.csv',index=False)
    text=['# R4 — rà từng thành phần và thử cách đơn giản hơn','',
        'Hoàn tất trên development: 8 GPU jobs có giám sát, 58 dòng policy/refiner trong hai suite (có lặp control giữa suite), 42 dòng dataset/policy/refiner khác nhau; 7 giả thuyết mới, không huấn luyện mạng mới. Hai cửa sổ Jetson đã xong, mỗi cửa sổ ba lần lặp trên mỗi dataset. Core, checkpoint, ngưỡng hành động và K34 giữ nguyên; chưa đưa ứng viên vào method chính.','',
        '## Kết luận','',
        'Có một tối ưu tốc độ không đổi quyết định ở nhánh bottleneck selector NAVSIM: dùng lại điểm cảnh gốc và chỉ tính thêm từng probe. Kết quả ghép cặp ở bảng dưới; mọi action/field/support được kiểm tra đúng nguyên trạng. Với chất lượng phanh, lượt này chưa tìm được cách vừa ít bỏ sót hơn, vừa ít phanh nhầm hơn trên cả hai dataset. Các thay đổi có vẻ hợp lý trong mã cần được kiểm tra theo tần suất tác động thực tế: ghép box thận trọng hơn, phủ union và kiểm tra nhóm loại vật thể không giải quyết phần lớn lỗi hiện tại.',
        'Field dạng đóng và support có vai trò rõ, đã tương đối gọn. Điểm yếu tập trung ở chất lượng/trạng thái của bằng chứng temporal, sự phụ thuộc của range–lateral–TTC và tác động của refiner lên quyết định. Đánh giá chi tiết 13 thành phần: [COMPONENT_REVIEW.md](COMPONENT_REVIEW.md).','',
        '- **Trung vị theo vật thể, không mạng mới:** AV2 bỏ sót 49→37/366; loss thiếu phanh 0.36851→0.31515 (−14.5%; delta −0.05336 [−0.07145,−0.03851]). Có 12 episode được cứu, không episode mới bị mất; 9/12 có bằng chứng từ vật thể quyết định, không ca cứu mới nào có peak nhu cầu HARD. Đổi lại false episodes/1.000 tăng 9.001→11.578 và false frames/1.000 tăng 53.29→108.94. NuScenes vẫn 38/99; delta loss +0.00360 [−0.00192,+0.01197]. Đây là tradeoff, không thay thế chung đã được chứng minh.',
        'W1 chưa được xóa: median vẫn bỏ sót 37/366 trên AV2 so với cheap 30/366, và 38/99 trên nuScenes so với cheap 34/99. Các mức phanh nhầm khác nhau, nên không diễn giải đây là so sánh tại cùng operating point.',
        '- **Chỉ bảo vệ HARD khi cheap và raw cùng đồng ý:** không cứu episode nào. AV2 giảm loss một ít nhưng G giảm −0.04606 [−0.06679,−0.02322]. Đếm episode phanh nhầm riêng lẻ gây hiểu lầm: episode giảm nhưng false-hard frames và chi phí phanh dư tăng.',
        '- **Một vòng + giữ extent:** pixel detector giảm 13.4% nuScenes / 16.5% AV2. NuScenes loss tăng +0.03222 [0.00325,0.08471], G giảm −0.06494 [−0.13344,−0.01727]. Có cả episode mới mất; trên AV2 chỉ 1/4 episode cứu có bằng chứng từ vật thể quyết định. Không chọn làm cấu hình chung.',
        '- **Containment / union / semantic consistency:** không thêm episode được cứu. Exact union chỉ giảm pixel khoảng 0.08%/0.27%; thay đổi còn lại gần như không tác động. Không gọi những sửa đổi này là cải thiện chất lượng đã chứng minh.',
        '- **Bỏ tinh chỉnh khi hai phiếu đã quyết định:** bỏ được 74.53% [65.90,80.86] đối tượng refiner ở nuScenes và 67.09% [63.68,70.60] ở AV2 trong policy trung vị. Replay đủ 29.782 frame không đổi action. Đây là tương đương với policy trung vị mới, không tương đương v6-S gốc. Bản triển khai đầu có chi phí đối chiếu dư; cửa sổ thứ hai bỏ tính lặp và không đối chiếu ở frame không có support cần tinh chỉnh. Bảng runtime dưới đây quyết định lợi ích thực tế.',
        '- R3 **critical-extent persistence** vẫn là bổ sung đơn giản có bằng chứng giảm loss với overhead nhỏ; R3 **point + extent** có đánh đổi chất lượng cân bằng hơn một số đầu học phức tạp. Chưa có lý do thay chúng bằng một ứng viên R4 như một cải thiện toàn diện. Không tăng độ phức tạp chỉ để tối ưu G.','',
        '## Phạm vi và cách đọc','',
        'nuScenes: 152 development scenes, 6.117 frames, 99 required episodes; AV2: 300 CFG-DEV logs, 23.665 masked frames, 366 episodes. Thống kê paired unit bootstrap 2.000, seed 0, CI 95% pointwise; nhiều lượt khám phá thích nghi, chưa có xác nhận độc lập. Ngưỡng DECELERATE/HARD giữ 1/3.5 m/s². Onset delay tính cả thời lượng episode nếu bị bỏ sót hoàn toàn. G giữ quy ước jerk theo hành động cheap của frame trước.',
        'So sánh chính với round1 B2K-A+recovery, không gán toàn bộ chênh lệch với frozen v6-S cho R4. Loss thiếu phanh là bình phương phần demand vượt commanded braking; không phải xác suất tai nạn hay kết quả closed-loop. False episode và false frame là hai lượng khác nhau.',
        'Readout CPU đổi duy nhất quyết định cuối trên các scene raw/v3a đã lưu; acquisition/history round1 không phụ thuộc quyết định này. Bốn policy acquisition đều chạy inference lại với history riêng. Không diễn giải replay readout như chạy lại một selector mới.',
        'Các cột `/none`, `/half`, `/speed`, `/speedhalf` giữ nguyên như control. Mọi dòng ở [all_rows.csv](all_rows.csv); mọi CI và effect ở [all_intervals.csv](all_intervals.csv), [all_contrasts.csv](all_contrasts.csv).','',
        '## Kết quả chính','',md(table),'','## Hiệu ứng ghép cặp so với round1','',md(effect),'',
        '## Selector NAVSIM: rút gọn chính xác bằng phép minimum','',
        'Extension riêng, cố định trước khi đo: cùng 1.000 token / 125 unit của K34/K20 trên navtest, chỉ kiểm tra runtime/parity post-hoc; không chọn chất lượng trên test. K31 giữ 15 trajectory và cùng h; s(S+probe)=min(s(S),h(probe)) nên không cần tính lại toàn bộ vật thể cũ. Ba lần lặp trên cùng CPU VUW1, 6.000 phép so sánh action/field/support đúng hoàn toàn. Không đổi planner, không suy ra speedup PDM-Closed hay toàn pipeline ảnh.','',md(navtable),'',md(navratio),'','[Báo cáo selector](nav_selector/REPORT.md), CSV, CI và hash nguồn giữ riêng.','',
        '## Tốc độ Jetson: tất cả các dòng của cả hai cửa sổ','',
        'Mỗi lần đo: 200 frame warm-up + 512 frame timed/dataset, ba lần lặp; 13 nuScenes scenes / 8 AV2 CFG-DEV logs trong đoạn timed. trt32, MAXN, khóa clocks, fan255, độc quyền, thứ tự pipeline luân phiên; không build engine mới. Mỗi pipeline tự có history và chạy detection riêng. Không suy tốc độ từ số pixel hoặc số vật thể CNN.',
        'So sánh paired trong từng cửa sổ; không lấy hiệu số hai mean khác cửa sổ làm bằng chứng speedup. Full median và skip median phải cùng action ở mọi frame warm-up/timed. Cả hai vẫn có tradeoff chất lượng của median.','',md(lt),'',md(le),'',
        'Bản trực tiếp: [jetson/REPORT.md](jetson/REPORT.md); bản bỏ tính lặp: [jetson_fast/REPORT.md](jetson_fast/REPORT.md). CI cho mọi tail/threshold trong từng thư mục. [jetson_backend/REPORT.md](jetson_backend/REPORT.md) giữ lại mọi khác biệt action giữa VUW và Jetson; không thay full-population quality bằng đoạn runtime nhỏ.','',
        '## Cơ chế và các case','',md(mech),'',
        'Cross-group history links chỉ khoảng 0.22–0.24%; đây là chẩn đoán nhóm loại, không đo đúng/sai identity. Vì vậy sửa class gate không giải quyết các chuyển nhầm giữa hai vật thể cùng loại. Các lỗi cùng loại đã được kiểm tra trực tiếp trong R2/R3 vẫn còn liên quan.',
        'Ví dụ AV2 0b97… frame75: demand của vật thể quyết định raw 1.040, cheap 1.087, refiner 0.889, GT 1.009. Trung vị phục hồi DECELERATE; frame76–77 mất cheap match nên vẫn không cứu được cả đoạn. Ví dụ 286170… frame35: scale làm mép lateral từ −1.095 thành −1.207 m, vừa ra khỏi corridor ±1.2, khiến demand 1.977→0; trung vị cứu được. Đây là trường hợp boundary cụ thể, không chứng minh lateral-only correction phù hợp chung.',
        'Ví dụ 14bf… frame51: median phanh nhưng matched decisive object vẫn demand 0. Ghi nhận episode được cứu mà không quy công cho đúng đối tượng. Không có bằng chứng matched decisive object không đồng nghĩa mọi đối tượng khác đều vô hại.',
        'Tất cả timeline của 12 episode median cứu ở mechanisms/median_rescue_timelines.csv; tất cả episode đổi ở dev/changed_episode_attribution.csv và readout/changed_episode_attribution.csv. Framewise median không có attribution theo vật thể; các ô tương ứng để trống. Các hình minh họa số liệu ở quality_tradeoff.pdf/png và case_timelines.pdf/png.','',
        '## Tái lập và trạng thái','',
        'Protocol và scripts: experiments/optimization_r4_20261024/. Trên VUW/VUW1 luôn source /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh; GPU dùng nguyên wrapper run_when_gpu_free.sh, SHARE=1 MIN_FREE_MB=10000, hai worker/group. Lệnh, exit và hash từng job ở job_status/. CPU: analyze.py --stage dev/readout; audit_rescues.py --stage <stage> --report <stage>; audit_mechanics.py; cost_audit.py. Jetson: window.py trong hai thư mục R4 Jetson; report_jetson.py --variant direct/fast. Tổng hợp: scripts/py experiments/optimization_r4_20261024/consolidate.py.',
        'Snapshot main-source trước/sau, zero baseline replay difference, parity shortcut, repetition stability và trả lại hardware nằm trong completion_audit.json. Mọi CSV/PDF/PNG và báo cáo được mirror ở results/vuw_summaries/OPTIMIZATION_R4_20261024/. Những điểm bất lợi được giữ nguyên. Không có candidate được promote, không tuning trên test.']
    (OUT/'TONG_HOP.md').write_text('\n'.join(text)+'\n')
    fast_effect=le[le.window=='jetson_fast']
    report=['# R4 — component simplification study (complete)','',
        '8 supervised GPU jobs exited 0; 58 suite-labelled development rows (42 unique dataset/policy/refiner rows), all 452 units; two completed Jetson windows, three repetitions/dataset each. No core, weight, threshold or K34 change; no test selection or promotion. [Vietnamese synthesis](TONG_HOP.md), [component review](COMPONENT_REVIEW.md), [exact simplification notes](SIMPLIFICATION_NOTES.md), [all rows](all_rows.csv), [all paired effects](all_contrasts.csv).','',
        'Paired scene/log bootstrap 2,000 seed0, pointwise 95%; adaptive exploratory development. Main comparator is round1 B2K-A+recovery. nuScenes 152 scenes/6,117 frames/99 episodes; masked AV2 CFG-DEV 300 logs/23,665 frames/366 episodes.','',
        '**Exact runtime improvement:** K31 bottleneck selector caches the base object scores and batches probe-only columns. All action/field/support outputs match in 6,000 paired comparisons on K34’s fixed 1,000 navtest tokens, three repetitions. Post-hoc timing only; no quality selection. Same-host selector-stage speedups:','',md(navratio),'','Details and absolute costs: [nav_selector/REPORT.md](nav_selector/REPORT.md). This is not a general PDM-Closed or end-to-end perception speedup.','',
        '- **Simple objectwise median:** AV2 misses 49→37, loss delta −0.05336 [−0.07145,−0.03851], false episodes +2.578 [1.269,3.976]/1,000; false frames 53.29→108.94/1,000. 9/12 rescues have decisive-object evidence, none is a new hard-peak rescue. nuScenes misses unchanged, loss +0.00360 [−0.00192,0.01197]. A tradeoff, not common dominance.',
        '- **One-round extent:** fewer requested pixels (−13.4%/−16.5%), but nuScenes loss +0.03222 [0.00325,0.08471] and G −0.06494 [−0.13344,−0.01727]. New misses and weak AV2 rescue attribution; reject as a common simplification.',
        '- **Containment / union / semantic checks:** no additional episode rescue; union pixel saving only ~0.08%/0.27%. Hard-brake corroboration worsens AV2 G −0.04606 [−0.06679,−0.02322] despite fewer false episodes.',
        '- **Exact abstention within median policy:** 74.53% [65.90,80.86] / 67.09% [63.68,70.60] refiner objects avoidable; all 29,782 stored-scene actions match. Direct implementation overhead motivated removal of duplicate matching/demand work. Both runtime implementations retained; do not confuse median-policy parity with original-method parity.','',
        'Optimized implementation mean-latency paired effects:','',md(fast_effect),'',
        'All latency tails and three repetitions: [direct](jetson/REPORT.md), [optimized](jetson_fast/REPORT.md); [backend differences](jetson_backend/REPORT.md). R3 extent persistence remains the simplest supported quality addition; R4 does not remove the cross-dataset tradeoff. Severe misses remain.','',
        'Exact commands/code hashes: experiments/optimization_r4_20261024/job_status/ and two R4 Jetson window logs. Reproduce synthesis: `scripts/py experiments/optimization_r4_20261024/consolidate.py`. [Completion audit](completion_audit.json), [execution notes](execution_notes.md), [quality figure](quality_tradeoff.pdf), [cases](case_timelines.pdf). Local mirror: results/vuw_summaries/OPTIMIZATION_R4_20261024/.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    shutil.copy2(HERE/'protocol.md',OUT/'protocol.md');shutil.copy2(HERE/'execution_notes.md',OUT/'execution_notes.md')
    make_figures(key)
    audit=json.loads((OUT/'server_completion_audit.json').read_text());assert audit['status']=='complete' and audit['all_exit_zero']
    frozen=json.loads((HERE/'fork_manifest.json').read_text());changed=[p for p,h in frozen.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h];assert not changed,changed
    windows={}
    for folder in ['optimization_r4_jetson_20261024','optimization_r4_fast_jetson_20261024']:
        state=json.loads((ROOT/'experiments'/folder/'logs/state.json').read_text());assert state['status']=='complete' and state['window_released'] and not state['restore_errors']
        assert len([r for r in state['runs'] if not r['smoke']])==6
        windows[folder]=dict(status='complete',timing_runs=6,source_engine_hashes_unchanged=state['source_engine_hashes_unchanged'],window_released=True)
    for folder in ['jetson','jetson_fast']:
        assert pd.read_csv(OUT/folder/'abstention_parity.csv').median_abstention_action_differences.sum()==0
        assert pd.read_csv(OUT/folder/'repetition_parity.csv').differing_action_groups.sum()==0
    assert json.loads((OUT/'mechanisms/audit_DONE.json').read_text())['abstention_action_differences']==0
    assert json.loads((OUT/'jetson_backend/DONE.json').read_text())['status']=='complete'
    audit.update(status='complete',local_source_files=len(frozen),local_sources_unchanged=True,full_development_rows=len(full),unique_rows=42,
        jetson_windows=windows,stored_scene_abstention_action_differences=0,actual_jetson_abstention_action_differences=0,
        no_core_promotion=True,report_generated=time.time(),quality_test_data_used=False,posthoc_runtime_test_data_used=True,
        navsim_selector=navstate,summary_mirror='results/vuw_summaries/OPTIMIZATION_R4_20261024/')
    audit['test_data_used']=True
    (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/'STATUS.json').write_text(json.dumps(dict(phase='complete',updated=time.time(),full_development_rows=58,core_changed=False),indent=2)+'\n')
    mirror=ROOT/'results/vuw_summaries/OPTIMIZATION_R4_20261024'
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in ('.md','.csv','.json','.pdf','.png'):
            dst=mirror/p.relative_to(OUT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    print('Complete:',OUT/'REPORT.md')

def make_figures(key):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    labels={'round1 combined':'Round1','critical extent retention':'Extent','round1 combined / half':'Half refiner',
        'round1 combined / none':'No refiner','one-round extent':'One round + extent','objectwise median':'Object median','hard-brake corroboration':'Hard vote'}
    contrasts=pd.read_csv(OUT/'all_contrasts.csv')
    names=[n for n in labels if n!='round1 combined']
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for j,ds in enumerate(['nusc','av2']):
        for i,(metric,title) in enumerate([('insufficient_loss_per_frame','Insufficient-braking loss / frame'),('false_braking_frames_per_1000','False-braking frames / 1,000')]):
            ax=axs[i,j]
            q=contrasts[(contrasts.dataset==ds)&(contrasts.comparator=='round1 combined')&(contrasts.metric==metric)&contrasts.pipeline.isin(names)].drop_duplicates('pipeline').set_index('pipeline').loc[names]
            ax.errorbar(q.candidate_minus_baseline,np.arange(len(q)),xerr=np.vstack([q.candidate_minus_baseline-q.ci_lo,q.ci_hi-q.candidate_minus_baseline]),fmt='o',capsize=3)
            ax.axvline(0,color='black',lw=.8,ls='--');ax.set_yticks(np.arange(len(q)));ax.set_yticklabels([labels[n] for n in names],fontsize=9)
            ax.set_title(ds+' — '+title,fontsize=10);ax.set_xlabel('Candidate minus round1 (lower is better)');ax.grid(axis='x',alpha=.2)
    axs[0,0].invert_yaxis();fig.suptitle('Quality tradeoffs: paired unit 95% intervals, development only');fig.tight_layout()
    fig.savefig(OUT/'quality_tradeoff.pdf');fig.savefig(OUT/'quality_tradeoff.png',dpi=160);plt.close(fig)
    timeline=pd.read_csv(OUT/'mechanisms/median_rescue_timelines.csv')
    cases=[('0b97f5dd-c396-3c02-b07d-b7fdbcb6c3d0','Range correction crosses threshold'),('28617035-7557-3cb9-99c2-754f72fd34b4','Lateral extent exits corridor'),('14bf638b-8f0d-35b2-a369-6d846b5b3892','Episode rescue without decisive-object evidence')]
    fig,axs=plt.subplots(3,1,figsize=(9,8))
    for ax,(unit,title) in zip(axs,cases):
        g=timeline[timeline.unit==unit]
        for c,l in [('gt_demand','GT scene'),('decisive_raw_demand','Decisive raw'),('decisive_refined_demand','Decisive refined'),('decisive_median_demand','Decisive median')]:ax.plot(g.frame,g[c],'o-',label=l,alpha=.8)
        ax.axhline(1,color='black',ls='--',lw=.8);ax.set_title(title+' — '+unit[:8],fontsize=10)
        ax.set_xlabel('Frame');ax.set_ylabel('Demand (m/s²)');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'case_timelines.pdf');fig.savefig(OUT/'case_timelines.png',dpi=160);plt.close(fig)
    nav=pd.read_csv(OUT/'nav_selector/summary.csv');fig,ax=plt.subplots(figsize=(6,4))
    for j,method in enumerate(['original','cached']):
        q=nav[(nav.method==method)&(nav.metric=='mean_total_ms')].set_index('grid').loc[['7x7','3x3']]
        ax.bar(np.arange(2)+(j-.5)*.35,q.estimate,.35,label=method,yerr=np.vstack([q.estimate-q.ci_lo,q.ci_hi-q.estimate]),capsize=3)
    ax.set_xticks([0,1]);ax.set_xticklabels(['7×7','3×3']);ax.set_ylabel('Selector-stage mean milliseconds');ax.legend();ax.grid(axis='y',alpha=.2)
    ax.set_title('Exact bottleneck score reuse — same-host CPU');fig.tight_layout()
    fig.savefig(OUT/'nav_selector/cost.pdf');fig.savefig(OUT/'nav_selector/cost.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
