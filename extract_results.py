import pickle, os

results_dir = 'results'

for label, fname in [('Single α=0.25', 'single_alpha0.25.pkl'),
                      ('Single α=0.5', 'single_alpha0.5.pkl')]:
    path = os.path.join(results_dir, fname)
    if not os.path.exists(path):
        continue
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f'=== {label} (baseline={data[list(data.keys())[0]]["baseline"]:.1f}) ===')
    # Find the T=10000 key
    t10000_key = None
    for k in data.keys():
        if k.startswith('T=10000'):
            t10000_key = k
            break
    if t10000_key:
        config_data = data[t10000_key]
        algos = config_data['algorithms']
        for algo_name in ['TS-fixed', 'TS-update', 'BZ', 'PD-BwK', 'TS (unconstrained)']:
            if algo_name in algos:
                res = algos[algo_name]
                print(f'  {algo_name}: mean={res["mean_revenue"]:.1f}, std={res["std_revenue"]:.1f}, %opt={res["mean_pct_optimal"]:.2f}')
    print()

for label, fname in [('Multi Linear', 'multi_linear.pkl'),
                      ('Multi Exponential', 'multi_exponential.pkl'),
                      ('Multi Logit', 'multi_logit.pkl')]:
    path = os.path.join(results_dir, fname)
    if not os.path.exists(path):
        continue
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f'=== {label} ===')
    # Find the T=10000 key
    t10000_key = None
    for k in data.keys():
        if k.startswith('T=10000'):
            t10000_key = k
            break
    if not t10000_key:
        # Try largest T
        max_t = max(int(k.split('_')[0].replace('T=','')) for k in data.keys())
        t10000_key = [k for k in data.keys() if k.startswith(f'T={max_t}')][0]
    if t10000_key:
        config_data = data[t10000_key]
        algos = config_data['algorithms']
        baseline = config_data.get('baseline', None)
        for algo_name in ['TS-fixed', 'TS-update', 'BZ']:
            if algo_name in algos:
                res = algos[algo_name]
                print(f'  {algo_name}: mean={res["mean_revenue"]:.1f}, std={res["std_revenue"]:.1f}, %opt={res["mean_pct_optimal"]:.2f}')
    print()