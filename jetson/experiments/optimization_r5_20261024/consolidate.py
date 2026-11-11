"""Complete R5 handoff from every executed row; do not promote a configuration."""
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R5_20261024'
MIRROR=ROOT/'results/vuw_summaries/OPTIMIZATION_R5_20261024'
BASE='round1 combined'
PRIOR='point fallback + extent retention'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,default=str)+'\n')
def table(d):
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |',
        *['| '+' | '.join(map(str,r))+' |' for r in d.itertuples(index=False,name=None)]])
def copytree(src,dst):
    for p in src.rglob('*'):
        if p.is_file() and p.suffix in ('.md','.csv','.json','.pdf','.png','.log'):
            q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)

def figures(summary,contrasts):
    colors={'R5 both':'tab:blue','R5 residual_both':'tab:orange','R5 point_residual':'tab:green','R5 cond point conditional':'tab:purple'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,ds in zip(axes,['nusc','av2']):
        d=summary[summary.dataset==ds];ax.scatter(d.false_braking_frames_per_1000,d.insufficient_loss_per_frame,c='.8',s=12,label='Every executed row')
        for prefix,color in colors.items():
            q=d[d.pipeline.isin([prefix+' q'+str(x) for x in [50,80,95]])]
            ax.plot(q.false_braking_frames_per_1000,q.insufficient_loss_per_frame,'o-',color=color,label=prefix.replace('R5 ',''),ms=4)
        for name,marker,color,label in [(BASE,'s','black','round1'),(PRIOR,'D','red','R3 point+extent'),('objectwise median','^','brown','R4 median')]:
            r=d[d.pipeline==name].iloc[0];ax.scatter([r.false_braking_frames_per_1000],[r.insufficient_loss_per_frame],s=60,marker=marker,color=color,label=label)
        ax.set_title(ds+' — development only');ax.set_xlabel('False-braking frames / 1,000 frames');ax.set_ylabel('Insufficient-braking loss / frame');ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,loc='upper right');fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(OUT/('quality_tradeoff.'+ext),dpi=160)
    plt.close(fig)
    names=['R5 both q50','R5 residual_both q50','R5 point_residual q50','R5 point_residual q80','R5 cond point conditional q50']
    fig,axes=plt.subplots(2,2,figsize=(12,6.2))
    for row,ds in enumerate(['nusc','av2']):
        for col,metric in enumerate(['insufficient_loss_per_frame','false_braking_frames_per_1000']):
            ax=axes[row,col]
            for i,name in enumerate(names):
                r=contrasts[(contrasts.dataset==ds)&(contrasts.pipeline==name)&(contrasts.comparator==PRIOR)&(contrasts.metric==metric)].iloc[0]
                ax.plot([r.ci_lo,r.ci_hi],[i,i],color='tab:blue');ax.plot(r.candidate_minus_baseline,i,'o',color='tab:blue')
            ax.axvline(0,color='.4',ls='--');ax.set_yticks(range(len(names)));ax.set_yticklabels([x.replace('R5 ','') for x in names],fontsize=8)
            ax.set_title(ds+' — delta versus R3 point+extent');ax.set_xlabel('Loss / frame' if col==0 else 'False-braking frames / 1,000');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Paired unit bootstrap 2,000, seed 0; pointwise exploratory 95% intervals');fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(OUT/('incremental_effects.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,ds in zip(axes,['nusc','av2']):
        d=pd.read_csv(OUT/'audit'/(ds+'_signatures.csv'));d=d[d.scope=='missed_braking'].sort_values('loss_share')
        ax.barh(d.signature.str.replace('_',' '),100*d.loss_share);ax.set_xlim(0,55);ax.set_xlabel('% loss within unbraked required frames');ax.set_title(ds+' — signatures, not causal fractions');ax.tick_params(axis='y',labelsize=7)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(OUT/('failure_signatures.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,['nusc','av2']):
        u=pd.read_csv(OUT/'survey'/ds/'unit_metrics.csv');b=u[u.pipeline_column=='a_v6S_b2A_missing_extent_retro|v3a'].set_index('unit')
        c=u[u.pipeline_column=='a_v6S_r5_point_residual_50|v3a'].set_index('unit');gain=(b.risk-c.risk).sort_values(ascending=False);gain=gain[gain>1e-10]
        ax.bar(range(1,len(gain)+1),100*gain/gain.sum());ax.set_xlabel('Improved unit, ranked by loss reduction');ax.set_ylabel('% of incremental risk reduction');ax.set_title(ds+' — point+extent + residual q50');ax.set_xticks(range(1,len(gain)+1))
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(OUT/('gain_concentration.'+ext),dpi=160)
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True);copytree(HERE/'reports',OUT)
    copytree(HERE/'model_reports',OUT/'training');copytree(HERE/'conditional_model_reports',OUT/'conditional_training')
    copytree(HERE/'job_status',OUT/'job_status');copytree(HERE/'protocol_versions',OUT/'protocol_versions')
    shutil.copy2(HERE/'protocol.md',OUT/'protocol.md')
    datasets=['nusc','av2'];summary=pd.concat([pd.read_csv(OUT/('final_'+d)/'summary.csv') for d in datasets],ignore_index=True)
    contrasts=pd.concat([pd.read_csv(OUT/('final_'+d)/'contrasts.csv') for d in datasets],ignore_index=True)
    intervals=pd.concat([pd.read_csv(OUT/('final_'+d)/'intervals.csv') for d in datasets],ignore_index=True)
    assert len(summary)==104 and set(summary.units)=={152,300}
    assert summary.groupby('dataset').size().to_dict()=={'av2':52,'nusc':52}
    assert not summary.duplicated(['dataset','pipeline']).any()
    summary.to_csv(OUT/'all_rows.csv',index=False);contrasts.to_csv(OUT/'all_contrasts.csv',index=False);intervals.to_csv(OUT/'all_intervals.csv',index=False)
    choices=['cheap','v6-S',BASE,PRIOR,'objectwise median','R5 both q50','R5 both q80','R5 residual ungated',
        'R5 residual_both q50','R5 extent_both q50','R5 point_residual q50','R5 point_residual q80','R5 cond point conditional q50']
    key=summary[summary.pipeline.isin(choices)].copy();key.to_csv(OUT/'key_results.csv',index=False)
    effects=contrasts[contrasts.pipeline.isin(choices)&contrasts.comparator.isin([BASE,PRIOR])&contrasts.metric.isin([
        'G','all_missed_rate','false_per_1000','false_braking_frames_per_1000','false_hard_frames_per_1000','overbraking_cost_per_frame','insufficient_loss_per_frame'])].copy()
    effects.to_csv(OUT/'key_effects.csv',index=False)
    kt=key[['dataset','pipeline','missed','episodes','false_per_1000','false_braking_frames_per_1000','insufficient_loss_per_frame','G']].copy()
    kt['missed/episodes']=kt.missed.astype(str)+'/'+kt.episodes.astype(str);kt=kt.drop(columns=['missed','episodes'])
    for col in ['false_per_1000','false_braking_frames_per_1000','insufficient_loss_per_frame','G']:kt[col]=kt[col].map(lambda x:f'{x:.5f}')
    ef=effects[effects.pipeline.isin(['R5 point_residual q50','R5 point_residual q80','R5 cond point conditional q50'])&(effects.comparator==PRIOR)].copy()
    ef['delta [95% CI]']=ef.apply(lambda r:f'{r.candidate_minus_baseline:+.5f} [{r.ci_lo:+.5f}, {r.ci_hi:+.5f}]',axis=1)
    eft=ef[['dataset','pipeline','metric','delta [95% CI]']]
    local=json.loads((HERE/'local_sources_before.json').read_text());changed=[p for p,h in local.items() if sha(p)!=h];assert not changed,changed
    save(OUT/'local_sources_before.json',local);save(OUT/'local_sources_after.json',{p:sha(p) for p in local})
    jobs=[]
    for p in sorted((HERE/'job_status').glob('*.json')):
        if 'sources' in p.name:continue
        state=json.loads(p.read_text());assert state['status']=='complete' and state['exit_code']==0 and not state['changed_main_sources'],state
        before=json.loads(p.with_name(p.stem+'.sources_before.json').read_text());after=json.loads(p.with_name(p.stem+'.sources_after.json').read_text());assert before==after
        jobs.append(dict(task=state['task'],dataset=state['dataset'],host=state['host'],seconds=state['finished']-state['started'],exit_code=0,frozen_files=len(before)))
    assert len(jobs)==23;pd.DataFrame(jobs).to_csv(OUT/'jobs.csv',index=False)
    attempts=[]
    for p in (HERE/'job_status/attempts').glob('*.json'):
        d=json.loads(p.read_text());attempts.append(dict(file=p.name,task=d['task'],dataset=d['dataset'],status=d['status'],exit_code=d.get('exit_code')))
    assert len(attempts)==2 and all(d['task']=='evaluate' and d['exit_code']==1 for d in attempts)
    for ds in datasets:
        parity=pd.read_csv(OUT/('final_'+ds)/'baseline_parity.csv');assert len(parity)=={'nusc':152,'av2':300}[ds] and parity.baseline_differing.sum()==0
        for folder,localfolder in [('training','models'),('conditional_training','conditional_models')]:
            report=json.loads((OUT/folder/ds/'REPORT.json').read_text())
            for m in report['models']:assert sha(HERE/localfolder/ds/(m['model']+'.pkl'))==m['sha256']
    code={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.glob('*.py'))}
    save(OUT/'code_sha256.json',code)
    audit=dict(status='complete',date='2026-10-25',started='2026-10-24',supervised_jobs=23,completed_jobs=jobs,retained_failed_attempts=attempts,
        development_rows=104,new_readout_configurations=44,controls_per_dataset=8,datasets={'nusc':dict(units=152,frames=6117,episodes=99),'av2':dict(units=300,frames=23665,episodes=366)},
        all_rows_retained=True,baseline_action_cost_parity_differences=0,stored_prior_v3a_trace_action_differences=0,
        paired_bootstrap=2000,seed=0,interval_scope='pointwise exploratory; adaptive development',
        local_source_files=len(local),local_sources_unchanged=True,remote_frozen_source_weights_unchanged=True,
        original_training_nonempty_units=783,trained_small_classifiers=8,quality_test_data_used=False,core_promoted=False,
        new_jetson_window=False,new_latency_measurement=False,
        scope='Final-readout experiments on stored acquisitions; original geometry/refinement still drives any subsequent state. No new detector acquisition or closed-loop evidence.')
    save(OUT/'completion_audit.json',audit);figures(summary,contrasts)
    report='''# R5 — conditional decision repair (complete, 2026-10-25)

**A modest decision improvement, not a general solution.** 23 supervised jobs completed; 104 full-development rows (44 new readout configurations × 2 datasets + 16 controls), eight small classifiers, nine inspected image panels. Original train/REF-TRAIN and unit-disjoint holdout only; evaluation on 152 nuScenes scenes / 300 masked AV2 CFG-DEV logs. Core, weights, thresholds and K34 unchanged. Every row is retained in [all_rows.csv](all_rows.csv); all intervals and contrasts are linked below.

The useful incremental result is retaining current cheap evidence **only when** its box was covered by an executed crop, no compatible final detection remains, and a training-only classifier corroborates its action. Added to the prior R3 point+extent candidate at q=.5:

| Dataset | Missed episodes, R3→new | Loss/frame, R3→new | Paired loss delta [95% CI] | False episodes/1,000 delta [95% CI] |
|---|---:|---:|---:|---:|
| nuScenes | 35→34 / 99 | .336997→.317066 | −.019931 [−.048364,−.000560] | +.326958 [0,.817929] |
| AV2 | 43→43 / 366 | .334890→.333299 | −.001591 [−.004150,−.000227] | +.211282 [.042126,.463824] |

This is −5.9% / −0.48% loss relative to R3, not the larger cumulative change versus frozen v6-S. False-braking frames/1,000 also increase: +.490436 [0,1.143796] / +.760617 [.210790,1.565301]. The additional nuScenes episode has decisive-object evidence. Gains occur in only 4 / 9 units; one unit contributes 52.1% / 67.9% of the loss reduction. At q=.8 the nuScenes loss reduction comes entirely from scene-1050 and its interval touches zero; no extra episode is rescued. No new hard-peak episode is rescued by the confidence-gated residual family.

**Negative/limiting outcomes:** keeping all residual cheap detections greatly increases false braking; AV2 ungated rescues have no matched decisive evidence. New all-object missing-history/rollback gates beat round1 but do not supersede R3 point+extent. Conditioning the classifier on raw/refiner disagreements adds little: on AV2 point+extent, one q=.5 row reduces misses 43→42, but that extra episode lacks matched decisive-object evidence. This is not a credible hard-miss fix. q is a model score, not guaranteed calibrated confidence.

**What the deeper audit changed:** "no decisive match" need not mean no visual detection. Truck fragments fail the frozen IoU match; some severe AV2 annotations touch the image edge or masked area. These are visibility/matching diagnostics, not grounds to discard annotations. All primary scores retain the original GT. Another 31.3–34.5% of required-frame loss occurs with an adequate discrete action; finite 2.5/6.0 m/s² commands can still leave shortfall. Do not equate every loss term with a selector failure.

All intervals: paired scene/log bootstrap, 2,000, seed 0, pointwise 95%, adaptive exploratory development. Upward-only corrections structurally reduce shortfall; success also requires false-braking and severity checks. G preserves K30's previous-cheap-action jerk convention. Final-readout replay does not change stored acquisition/history; it does not demonstrate a repaired geometric scene or extend the frozen selector theorem.

No new Jetson timing or core promotion: the limited, concentrated quality gains do not justify claiming deployment readiness. Detector requested pixels stay those of the source acquisition; CPU classifier overhead and cross-backend parity are unmeasured. [Mean/p95 pixels](detector_cost/REPORT.md), [full synthesis](TONG_HOP.md), [case review](CASE_REVIEW.md), [conditional calibration](CALIBRATION.md), [all paired effects](all_contrasts.csv), [all intervals](all_intervals.csv), [integrity audit](completion_audit.json).

Reproduce CPU tasks on VUW/VUW1 after `source /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh`: `python experiments/optimization_r5_20261024/job.py <cpu-task> <dataset>`. For the GPU task use local `python3 experiments/optimization_r5_20261024/launch_gpu.py VUW <new-group-name> '[["refine_train","nusc"],["refine_train","av2"]]'`; never launch it outside the GPU wrapper. Tasks and exact source hashes: [jobs.csv](jobs.csv), [code_sha256.json](code_sha256.json), [protocol](protocol.md). CPU tasks use one thread. Frozen-v3a training-crop inference used two workers behind unchanged `run_when_gpu_free.sh`, `SHARE=1 MIN_FREE_MB=10000`; no detector/CNN training. Local synthesis: `scripts/py experiments/optimization_r5_20261024/consolidate.py`. Summary mirror: `results/vuw_summaries/OPTIMIZATION_R5_20261024/`.
'''
    (OUT/'REPORT.md').write_text(report)
    synthesis=['# R5 — kết quả và giới hạn','',
        'Vòng này có cải thiện decision thực tế, nhưng chủ yếu là bổ sung nhỏ lên R3; chưa giải quyết được weakness chung trên cả hai bộ dữ liệu. Tất cả 104 dòng hoàn tất trên đủ 452 unit. 44 cấu hình readout mới/dataset, tám mô hình cây nhỏ được học từ train riêng. Không đổi core, checkpoint hay K34; không dùng test.',
        '## Phần mới có ích','',
        'Giữ lại có điều kiện một detection cheap hiện tại sau khi verification không còn detection tương ứng. Chỉ tác động khi tâm cheap nằm trong crop đã thực sự chạy, không có match final cùng nhóm ở IoU ≥0.3 và mô hình train xác nhận mức hành động. Không lấy box cũ làm vật thể hiện tại; không tự khôi phục mọi cheap false positive.',
        'Khi thêm vào R3 point+extent ở q=.5: nuScenes bỏ sót 35→34/99, loss .336997→.317066 (−5.9%); AV2 vẫn 43/366, loss .334890→.333299 (−0.48%). Phanh nhầm tăng tương ứng .327/.211 episode mỗi 1.000 frame. Bảng effect dưới đây so trực tiếp với R3, tránh quy phần đã cải thiện từ các vòng trước cho R5.',
        'Lợi ích tập trung: chỉ 4 scene nuScenes / 9 log AV2 có giảm loss. Scene-1050 chiếm 52.1% lợi ích nuScenes; một log AV2 chiếm 67.9%. Bỏ unit đóng góp lớn nhất chỉ là kiểm tra độ tập trung, không thay bảng chính. q=.8 nuScenes không tăng false braking ở các frame đã xét, nhưng toàn bộ lợi ích đến từ một scene và CI chạm 0; chưa đủ bằng chứng chọn làm mặc định.',
        '## Những hướng không đủ tốt','',
        'Bù thiếu lịch sử bằng cây nhẹ và rollback refiner có điều kiện: R5 both q=.5 giảm miss so với round1 từ 38→32 nuScenes và 49→46 AV2. Đổi lại false episodes tăng 2.289/1.394 mỗi 1.000 frame. So với R3 point+extent, loss ở nhóm HARD vẫn cao hơn; không nên thay một ứng viên tốt hơn chỉ vì miss count đẹp ở một bộ.',
        'Giữ cheap vô điều kiện: false episodes tăng rất mạnh, AV2 G giảm. Các episode được cứu bằng tín hiệu không từ vật thể quyết định phải được giữ trong báo cáo. Kết quả âm này là lý do cần gate, không phải lý do bỏ hàng đối chứng.',
        'Học riêng trên các object mà v3a hạ mức hành động: cải thiện calibration trên holdout nhưng lợi ích full development nhỏ. Dữ liệu conditional chỉ có 2 nhãn HARD ở nuScenes và 30 ở AV2; không kỳ vọng một mô hình nhỏ tự học chắc các ca hiếm đó. Dòng AV2 giảm 43→42 miss so với R3 không có bằng chứng matched decisive object cho episode mới; không gọi đó là xử lý được hard weakness.',
        '## Số liệu chính — mọi dòng còn lại ở all_rows.csv','',table(kt),'',
        '## Hiệu ứng mới so với R3 point+extent','',table(eft),'',
        '## Cách đánh giá và tái lập','',
        '152 nuScenes / 6.117 frame / 99 required episodes; 300 AV2 CFG-DEV / 23.665 masked frame / 366 episodes. Bootstrap ghép cặp theo scene/log, 2.000, seed 0, CI 95% pointwise; các giả thuyết mở rộng thích nghi được ghi theo thứ tự trong protocol.md. Đây là dev đã xem nhiều lần, không phải confirmatory test. Không chọn q bằng test và không giấu các hàng không có tác động.',
        'Các phương án chỉ sửa readout cuối. Refiner/geometry gốc vẫn điều khiển state, extent seed và acquisition của frame sau. Scalar evidence được chặn vào ngưỡng hành động của consumer; đây là sửa perception/decision readout thử nghiệm, không phải bằng chứng mọi tọa độ 3D đã tốt hơn. Thay đổi chỉ tăng mức hành động nên giảm risk tự thân chưa đủ: phải xem exposure phanh nhầm, excess cost, G và mức HARD.',
        'Không đo Jetson mới vì chưa có ứng viên chất lượng đủ mạnh để tuyên bố thay thế chung. Không suy tốc độ từ việc không thêm CNN hay từ số pixel không đổi. Khi cần tiếp tục triển khai, phải đo overhead cây/matching trên từng frame trong cửa sổ độc quyền, kiểm tra action parity và toàn bộ tails.',
        '23 job hoàn tất; hai lần chạy đầu của phần xuất thống kê severe gặp lỗi tên cột pandas `gt` trùng method. Đã sửa thành truy cập cột tường minh, chạy lại hai job đầy đủ; giữ log và failed status trong job_status/attempts. Không mất kết quả inference, không bỏ unit. Hash nguồn/checkpoint trước–sau không đổi. Không có job R5 còn dở trong phạm vi vòng này.',
        'Nguồn: REPORT.md, all_rows.csv, all_contrasts.csv, all_intervals.csv, final_nusc/ và final_av2/. Mọi episode thay đổi có attribution so với round1, R3 point+extent và extent riêng. survey/ giữ severity/geometry flags/concentration. training/ và conditional_training/ giữ provenance, holdout và model SHA256.']
    (OUT/'TONG_HOP.md').write_text('\n'.join(synthesis)+'\n')
    (OUT/'execution_notes.md').write_text('''# Execution notes

All 23 final jobs exited 0 with identical frozen-source/checkpoint snapshots.
The first evaluate_nusc/evaluate_av2 attempts failed only in additional severe
CSV export because `d.gt` resolved to a pandas method. Replaced with `d['gt']`,
reran both full evaluation jobs, and retained initial logs/status in
job_status/attempts/. No inference frames or negative outcomes were discarded.

The protocol was extended explicitly after development observations; it is not
a single untouched preregistration. protocol_versions/base.md and
protocol_versions/conditional.md reconstruct the exact append-only texts whose
SHA256 values were recorded at the respective training launches; hashes match.
All extensions and every confidence setting remain in the final protocol/table.

No Jetson reservation, TensorRT build, core promotion, or test-quality selection.
Stored acquisition replay is valid only for final-readout policies preserving
the original state update; it is not a fresh rollout with corrected geometry.
''')
    save(OUT/'STATUS.json',dict(phase='complete',updated=time.time(),development_rows=104,core_changed=False,all_jobs_complete=True))
    (OUT/'STATUS.md').write_text('# R5 complete\n\nAll 23 jobs and 104 development rows are complete. See [REPORT.md](REPORT.md). No core promotion or new Jetson latency claim.\n')
    # CASE_REVIEW and CALIBRATION are authored alongside this deterministic synthesis.
    for name in ['CASE_REVIEW.md','CALIBRATION.md','NEXT_STEPS.md']:
        if (HERE/name).exists():shutil.copy2(HERE/name,OUT/name)
    copytree(OUT,MIRROR)
    copied=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.md','.csv','.json','.pdf','.png','.log')]
    assert all(sha(p)==sha(MIRROR/p.relative_to(OUT)) for p in copied)
    save(HERE/'FINISH_STATUS.json',dict(status='complete',report=str(OUT/'REPORT.md'),mirrored_files=len(copied),mirror_hashes_verified=True,updated=time.time()))
    print('R5 complete:',OUT/'REPORT.md','rows',len(summary),'mirrored',len(copied))

if __name__=='__main__':main()
