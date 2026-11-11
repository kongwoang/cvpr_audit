"""Generate an isolated JP03 harness from the unchanged JP01 harness."""
import ast
from pathlib import Path
from build import once

HERE = Path(__file__).resolve().parent
text = (HERE / 'scripts/91_latency_opt.py').read_text()
text = once(text, '    args = ap.parse_args()\n', "    ap.add_argument('--c1_graph_mode', choices=['baseline','candidate','audit'], default='baseline')\n    args = ap.parse_args()\n")
text = once(text, '    import columns_only as CO\n', '    import columns_only as CO\n    import c1_fixed_graph as CG\n')
text = once(text, "    ADAPTER_ENCODER=ENC['trt32'] if args.adapter_encoder=='trt32' else CF.Encoder('t32',ENG,ckh)\n", """    ADAPTER_ENCODER=ENC['trt32'] if args.adapter_encoder=='trt32' else CF.Encoder('t32',ENG,ckh)
    assert args.columns_mode == 'baseline', 'JP03 isolates the C1 graph change'
    if args.c1_graph_mode != 'baseline':
        assert args.paths == 'trt32' and args.adapter_encoder == 'trt32'
        CG.install(ENC['trt32'], GR, audit=args.c1_graph_mode == 'audit')
""")
text = once(text, 'f"latency_opt_{args.columns_mode}_{args.graphs}', 'f"latency_c1_{args.c1_graph_mode}_{args.graphs}')
text = once(text, "        rec['debug_column_calls'] = column_checks[0]\n", "        rec['debug_column_calls'] = column_checks[0]\n        rec['debug_c1_graph_checks'] = CG.CHECKS[0]\n")
text = once(text, "args.columns_mode == 'audit' or args.record_history", "args.c1_graph_mode == 'audit' or args.columns_mode == 'audit' or args.record_history")
ast.parse(text)
(HERE / 'scripts/94_latency_c1_graph.py').write_text(text)
print('JP03 harness generated; no inference run.')
