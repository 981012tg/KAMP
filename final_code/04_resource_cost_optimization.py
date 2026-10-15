"""Chapter 4: month-wide, model-based production and cost simulation. Run from notebook 08."""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PEAK_LIMIT = 176.0


def setup(root, settings):
    root = Path(root)
    package = root / 'real_final/results/team_model_optimization_fixed176/_team_package_runtime/team_counterfactual'
    assert (package / 'counterfactual/cf_predict.py').exists(), '기존 팀원 반사실 예측 패키지가 필요합니다.'
    official = json.loads((root / 'results/preprocessing_24h/config.json').read_text(encoding='utf-8'))
    bundled = json.loads((package / 'results/preprocessing_24h/config.json').read_text(encoding='utf-8'))
    assert official['source_sha256'] == bundled['source_sha256'], '전처리 원본이 다릅니다.'
    sys.path.insert(0, str(package / 'counterfactual'))
    from cf_predict import CounterfactualPredictor

    cf = CounterfactualPredictor(verbose=True)
    raw = pd.read_csv(root / 'dataset/okm_augumented_2021.csv')
    raw['date'] = pd.to_datetime(raw['날짜'].astype(str), format='%Y%m%d')
    raw['hour'] = raw.groupby('date').cumcount()
    raw['origin'] = raw['date'] + pd.to_timedelta(raw['hour'], unit='h')
    month = pd.Period(settings['month'], freq='M')
    dates = pd.date_range(month.start_time, month.end_time.normalize())
    saved = pd.read_csv(
        root / 'results/team_submission_runtime/outputs/test_predictions_8_9월.csv',
        parse_dates=['기준시각', '예측대상시각'],
    )

    days = {}
    for date in dates:
        day_rows = raw.loc[raw.date.eq(date)].sort_values('hour')
        reference = saved.loc[saved['기준시각'].eq(date)].sort_values('예측칸')
        assert len(day_rows) == 24 and len(reference) == 96, f'{date}: 완전한 월 예측 필요'
        past = raw.loc[raw.date.lt(date)]
        capacity = past.groupby('hour')['생산량'].quantile(.95).reindex(range(24)).to_numpy(float)
        plan = cf.baseline_plan(date)
        np.testing.assert_allclose(plan, day_rows['생산량'].to_numpy(float))
        # 실제 설비 용량이 없으므로 과거 시간대별 생산량 상위 95%를 참고 한도로 사용한다.
        capacity = np.maximum(capacity, plan)
        days[date] = {
            'plan': plan,
            'capacity': capacity,
            'labor': day_rows['인건비'].to_numpy(float),
            'tariff': day_rows['전기요금(계절)'].to_numpy(float),
            'reference': reference,
        }
    return cf, days


def predict(cf, date, plan):
    """팀원 패키지와 같은 수치예측 및 LightGBM Q1 계산."""
    i = cf._locate(date)
    features = cf._features_for(i, plan)
    model_q, model_curve = cf.models(cf._week_start(date))
    q = np.maximum(model_q.predict(features[cf.W.L13]), 0).astype(np.float32)
    rule = features['rule3_oof'].to_numpy().astype(np.float32)
    residual_curve = np.maximum(
        model_curve.predict(features[cf.W.L13]) + rule, 0
    ).astype(np.float32)
    idle_gate = features['st_day_on'].to_numpy() == 0
    idle_profile = cf._idle_profile(i)
    curve = np.where(idle_gate, rule, residual_curve) if idle_profile is None else np.where(
        idle_gate & np.isfinite(idle_profile),
        idle_profile,
        np.where(idle_gate, rule, residual_curve),
    )
    assert np.isfinite(curve).all() and np.isfinite(q).all()
    return {'plan': np.asarray(plan, dtype=float).copy(), 'curve': curve, 'q': q}


def day_metrics(item, day, cfg, staff):
    curve = item['curve']
    plan = item['plan']
    energy_by_hour = curve.reshape(24, 4).sum(axis=1) * 0.25
    active = plan > cfg['active_epsilon']
    base_active = day['plan'] > cfg['active_epsilon']
    weighted_hours = float(day['labor'][active].sum())
    base_weighted_hours = float(day['labor'][base_active].sum())
    labor_won = staff * cfg['hourly_wage'] * weighted_hours
    added_labor_won = staff * cfg['hourly_wage'] * max(
        weighted_hours - base_weighted_hours, 0.0
    )
    q = item['q']
    return {
        'kwh': float(energy_by_hour.sum()),
        'peak_kw': float(curve.max()),
        'alarm_cells': int((q >= PEAK_LIMIT).sum()),
        'q_peak': float(q.max()),
        'q_excess': float(np.maximum(q - PEAK_LIMIT, 0).sum()),
        'usage_won': float(energy_by_hour @ day['tariff']),
        'labor_won': float(labor_won),
        'extra_labor_won': float(added_labor_won),
        'weighted_labor_hours': weighted_hours,
        'production': float(plan.sum()),
    }


def monthly(state, days, cfg, staff, basic_mode, baseline_peak):
    rows = [day_metrics(state[date], days[date], cfg, staff) for date in days]
    totals = {
        key: float(sum(row[key] for row in rows))
        for key in [
            'kwh', 'usage_won', 'labor_won', 'extra_labor_won',
            'weighted_labor_hours', 'production', 'alarm_cells', 'q_excess',
        ]
    }
    totals['peak_kw'] = max(row['peak_kw'] for row in rows)
    totals['q_peak'] = max(row['q_peak'] for row in rows)
    totals['billing_kw'] = baseline_peak if basic_mode == 'fixed' else totals['peak_kw']
    totals['basic_won'] = totals['billing_kw'] * cfg['basic_won_per_kw']
    totals['electricity_won'] = totals['basic_won'] + totals['usage_won']
    totals['total_won'] = totals['electricity_won'] + totals['labor_won']
    return totals


def objective_value(metrics, objective):
    if objective == 'electricity':
        return metrics['electricity_won']
    if objective == 'total':
        return metrics['total_won']
    raise ValueError(f'지원하지 않는 목적함수: {objective}')


def candidate_rank(metrics, objective):
    if objective == 'peak':
        return (
            round(metrics['peak_kw'], 6),
            round(metrics['electricity_won'], 2),
            round(metrics['total_won'], 2),
        )
    return (round(objective_value(metrics, objective), 2),)


def improves(trial, current, objective, cfg):
    if objective == 'peak':
        return trial['peak_kw'] < current['peak_kw'] - cfg['minimum_peak_improvement_kw']
    return objective_value(trial, objective) < (
        objective_value(current, objective) - cfg['minimum_improvement_won']
    )


def slot_pools(state, days, cfg, objective):
    sources = []
    destinations = []
    if objective == 'peak':
        source_pool_size = cfg['peak_source_pool_size']
        destination_pool_size = cfg['peak_destination_pool_size']
    elif objective == 'electricity':
        source_pool_size = cfg['electricity_source_pool_size']
        destination_pool_size = cfg['electricity_destination_pool_size']
    else:
        source_pool_size = cfg['source_pool_size']
        destination_pool_size = cfg['destination_pool_size']
    for date, item in state.items():
        plan = item['plan']
        risk = item['q'].reshape(24, 4).max(axis=1)
        power = item['curve'].reshape(24, 4).max(axis=1)
        for hour in range(24):
            amount = float(plan[hour])
            room = float(days[date]['capacity'][hour] - amount)
            labor = float(days[date]['labor'][hour])
            if amount > cfg['active_epsilon']:
                source_key = (
                    (-labor, amount, -risk[hour], -power[hour])
                    if objective == 'total'
                    # 기본요금은 월 최대전력으로 결정된다. 따라서 전기요금안은
                    # 경보값보다 현재 수치예측 전력이 큰 시간을 먼저 줄인다.
                    else (-power[hour], -risk[hour], amount)
                )
                sources.append((source_key, date, hour, amount))
            if room > cfg['active_epsilon']:
                already_active = amount > cfg['active_epsilon']
                if objective == 'total':
                    destination_key = (
                        0 if already_active else 1,
                        labor,
                        risk[hour],
                        power[hour],
                        -room,
                    )
                elif objective == 'peak':
                    destination_key = (
                        0 if already_active else 1,
                        power[hour],
                        risk[hour],
                        labor,
                        -room,
                    )
                else:
                    destination_key = (risk[hour], power[hour], -room, labor)
                destinations.append((destination_key, date, hour, room))

    sources.sort(key=lambda row: row[0])
    destinations.sort(key=lambda row: row[0])
    return (
        sources[:source_pool_size],
        destinations[:destination_pool_size],
    )


def candidate_moves(state, days, cfg, objective):
    sources, destinations = slot_pools(state, days, cfg, objective)
    for _, source_date, source_hour, source_amount in sources:
        for _, destination_date, destination_hour, destination_room in destinations:
            if source_date == destination_date and source_hour == destination_hour:
                continue
            seen_amounts = set()
            for fraction in cfg['move_fractions']:
                amount = min(source_amount * fraction, destination_room)
                amount_key = round(float(amount), 8)
                if amount <= cfg['active_epsilon'] or amount_key in seen_amounts:
                    continue
                seen_amounts.add(amount_key)
                yield (
                    source_date, source_hour,
                    destination_date, destination_hour,
                    float(amount),
                )


def peak_destination_orders(state, days, cfg, blocked_slots):
    rows = []
    for date, item in state.items():
        plan = item['plan']
        risk = item['q'].reshape(24, 4).max(axis=1)
        power = item['curve'].reshape(24, 4).max(axis=1)
        for hour in range(24):
            if (date, hour) in blocked_slots:
                continue
            room = float(days[date]['capacity'][hour] - plan[hour])
            if room <= cfg['active_epsilon']:
                continue
            rows.append({
                'date': date,
                'hour': hour,
                'room': room,
                'active': bool(plan[hour] > cfg['active_epsilon']),
                'power': float(power[hour]),
                'risk': float(risk[hour]),
                'labor': float(days[date]['labor'][hour]),
            })

    strategies = [
        lambda row: (not row['active'], row['power'], -row['room'], row['risk']),
        lambda row: (not row['active'], -row['room'], row['power'], row['risk']),
        lambda row: (row['power'], -row['room'], not row['active'], row['risk']),
        lambda row: (-row['room'], row['power'], not row['active'], row['risk']),
    ]
    scan_size = cfg['peak_destination_scan_size']
    return [sorted(rows, key=key)[:scan_size] for key in strategies]


def build_peak_batch(source_group, destination_order, fraction, cfg):
    blocked_slots = {(row[1], row[2]) for row in source_group}
    remaining_room = {
        (row['date'], row['hour']): row['room'] for row in destination_order
    }
    batch = []
    for _, source_date, source_hour, source_amount in source_group:
        remaining = source_amount * fraction
        leg_count = 0
        for destination in destination_order:
            destination_key = (destination['date'], destination['hour'])
            if destination_key in blocked_slots:
                continue
            room = remaining_room[destination_key]
            if room <= cfg['active_epsilon']:
                continue
            amount = min(remaining, room)
            if amount <= cfg['active_epsilon']:
                continue
            batch.append((
                source_date,
                source_hour,
                destination['date'],
                destination['hour'],
                float(amount),
            ))
            remaining_room[destination_key] -= amount
            remaining -= amount
            leg_count += 1
            if remaining <= cfg['active_epsilon']:
                break
            if leg_count >= cfg['peak_max_destination_legs']:
                break
    return batch


def peak_batch_candidates(state, days, cfg):
    sources, _ = slot_pools(state, days, cfg, 'peak')
    groups = [[source] for source in sources]
    max_group = min(cfg['peak_max_source_group'], len(sources))
    groups.extend([sources[:size] for size in range(2, max_group + 1)])
    seen = set()
    for source_group in groups:
        blocked_slots = {(row[1], row[2]) for row in source_group}
        orders = peak_destination_orders(state, days, cfg, blocked_slots)
        fractions = (
            cfg['peak_move_fractions']
            if len(source_group) == 1 else cfg['peak_group_fractions']
        )
        for destination_order in orders:
            for fraction in fractions:
                batch = build_peak_batch(
                    source_group, destination_order, fraction, cfg
                )
                if not batch:
                    continue
                signature = tuple(
                    (
                        str(source_date), source_hour,
                        str(destination_date), destination_hour,
                        round(amount, 6),
                    )
                    for source_date, source_hour, destination_date,
                    destination_hour, amount in batch
                )
                if signature in seen:
                    continue
                seen.add(signature)
                yield batch


def apply_moves(cf, state, moves):
    changed = dict(state)
    touched_dates = {
        date
        for move in moves
        for date in (move[0], move[2])
    }
    changed_plans = {
        date: state[date]['plan'].copy() for date in touched_dates
    }
    for source_date, source_hour, destination_date, destination_hour, amount in moves:
        changed_plans[source_date][source_hour] -= amount
        changed_plans[destination_date][destination_hour] += amount
    for date, plan in changed_plans.items():
        changed[date] = predict(cf, date, plan)
    return changed


def safety_ok(metrics, baseline_metrics, cfg):
    return (
        metrics['peak_kw'] <= baseline_metrics['peak_kw'] + cfg['safety_tolerance']
        and metrics['alarm_cells'] <= baseline_metrics['alarm_cells']
        and metrics['q_excess'] <= baseline_metrics['q_excess'] + cfg['safety_tolerance']
    )


def optimize(cf, baseline, days, cfg, objective, baseline_peak):
    state = dict(baseline)
    baseline_metrics = monthly(
        baseline, days, cfg, cfg['staff'], 'monthly_peak_assumption', baseline_peak
    )
    current_metrics = monthly(
        state, days, cfg, cfg['staff'], 'monthly_peak_assumption', baseline_peak
    )
    accepted_moves = []

    max_iterations = cfg['max_iterations_by_objective'][objective]
    for iteration in range(1, max_iterations + 1):
        best = None
        candidate_batches = (
            peak_batch_candidates(state, days, cfg)
            if objective == 'peak'
            else ([move] for move in candidate_moves(state, days, cfg, objective))
        )
        for moves in candidate_batches:
            trial_state = apply_moves(cf, state, moves)
            trial_metrics = monthly(
                trial_state, days, cfg, cfg['staff'],
                'monthly_peak_assumption', baseline_peak,
            )
            if not safety_ok(trial_metrics, baseline_metrics, cfg):
                continue
            if not improves(trial_metrics, current_metrics, objective, cfg):
                continue
            if best is None or candidate_rank(trial_metrics, objective) < candidate_rank(best[2], objective):
                best = (moves, trial_state, trial_metrics)

        if best is None:
            return state, current_metrics, accepted_moves, 'no_improvement_in_searched_candidates'

        moves, state, current_metrics = best
        for leg_number, move in enumerate(moves, start=1):
            source_date, source_hour, destination_date, destination_hour, amount = move
            accepted_moves.append({
                'objective': objective,
                'iteration': iteration,
                'batch_leg': leg_number,
                'batch_size': len(moves),
                'source_date': source_date,
                'source_hour': source_hour,
                'destination_date': destination_date,
                'destination_hour': destination_hour,
                'moved_production': amount,
                'electricity_won': current_metrics['electricity_won'],
                'labor_won': current_metrics['labor_won'],
                'total_won': current_metrics['total_won'],
                'peak_kw': current_metrics['peak_kw'],
            })
        moved_total = sum(move[4] for move in moves)
        print(
            objective, iteration,
            f'묶음 이동 {len(moves)}개',
            f'총 이동량={moved_total:.3f}',
            f'월 최대전력={current_metrics["peak_kw"]:.3f}',
            flush=True,
        )

    return state, current_metrics, accepted_moves, 'iteration_limit'


def run(root, cfg):
    root = Path(root)
    out = root / 'results/resource_cost_optimization_chapter4'
    out.mkdir(parents=True, exist_ok=True)
    cf, days = setup(root, cfg)

    baseline = {}
    checks = []
    for date, day in days.items():
        item = predict(cf, date, day['plan'])
        for column, key in [('curve_final', 'curve'), ('q1_new', 'q')]:
            error = float(np.abs(item[key] - day['reference'][column].to_numpy()).max())
            checks.append({'date': date, 'column': column, 'maximum_difference': error})
            assert error <= 0.002, f'{date} {column}: 02 최종 모델과 불일치 ({error})'
        baseline[date] = item
        print('원래 계획 확인:', date.date(), flush=True)
    pd.DataFrame(checks).to_csv(
        out / 'baseline_identity_checks.csv', index=False, encoding='utf-8-sig'
    )

    baseline_peak = max(float(item['curve'].max()) for item in baseline.values())
    chosen_states = {'baseline': baseline}
    stop_reasons = {'baseline': 'baseline'}
    all_moves = []
    for objective in ['peak', 'electricity', 'total']:
        state, _, moves, stop_reason = optimize(
            cf, baseline, days, cfg, objective, baseline_peak
        )
        chosen_states[objective] = state
        stop_reasons[objective] = stop_reason
        all_moves.extend(moves)

    baseline_production = sum(day['plan'].sum() for day in days.values())
    summaries = []
    schedules = []
    daily_rows = []
    prediction_rows = []
    sensitivity = []

    for objective, state in chosen_states.items():
        optimized_production = sum(item['plan'].sum() for item in state.values())
        assert np.isclose(optimized_production, baseline_production)
        for date, item in state.items():
            assert np.all(item['plan'] >= -cfg['safety_tolerance'])
            assert np.all(item['plan'] <= days[date]['capacity'] + cfg['safety_tolerance'])

        changed_days = sum(
            not np.allclose(item['plan'], days[date]['plan'])
            for date, item in state.items()
        )
        moved_production = 0.5 * sum(
            np.abs(item['plan'] - days[date]['plan']).sum()
            for date, item in state.items()
        )

        for basic_mode in ['fixed', 'monthly_peak_assumption']:
            current = monthly(state, days, cfg, cfg['staff'], basic_mode, baseline_peak)
            base = monthly(baseline, days, cfg, cfg['staff'], basic_mode, baseline_peak)
            row = {
                'basic_mode': basic_mode,
                'objective': objective,
                'optimization_basic_mode': 'monthly_peak_assumption',
                'staff': cfg['staff'],
                **current,
                'moved_production': moved_production,
                'changed_days': changed_days,
                'stop_reason': stop_reasons[objective],
            }
            for column in [
                'kwh', 'peak_kw', 'basic_won', 'usage_won', 'electricity_won',
                'labor_won', 'total_won', 'alarm_cells',
            ]:
                row[column + '_saving'] = base[column] - current[column]
            row['saving_percent'] = 100 * row['total_won_saving'] / base['total_won']
            summaries.append(row)

            for staff in cfg['staff_sensitivity']:
                for wage_factor in [1.0, 1.25, 1.5]:
                    sensitivity_cfg = {**cfg, 'hourly_wage': cfg['hourly_wage'] * wage_factor}
                    repriced = monthly(state, days, sensitivity_cfg, staff, basic_mode, baseline_peak)
                    repriced_base = monthly(baseline, days, sensitivity_cfg, staff, basic_mode, baseline_peak)
                    sensitivity.append({
                        'basic_mode': basic_mode,
                        'objective': objective,
                        'staff': staff,
                        'hourly_wage': sensitivity_cfg['hourly_wage'],
                        'saving_won': repriced_base['total_won'] - repriced['total_won'],
                    })

        for date, item in state.items():
            daily_rows.append({
                'objective': objective,
                'date': date,
                'baseline_production': float(days[date]['plan'].sum()),
                'optimized_production': float(item['plan'].sum()),
                'net_change': float(item['plan'].sum() - days[date]['plan'].sum()),
            })
            for hour in range(24):
                schedules.append({
                    'objective': objective,
                    'date': date,
                    'hour': hour,
                    'day_or_night': '주간' if 6 <= hour < 22 else '야간',
                    'labor_weight': days[date]['labor'][hour],
                    'capacity_reference': days[date]['capacity'][hour],
                    'baseline_production': days[date]['plan'][hour],
                    'optimized_production': item['plan'][hour],
                })
            for interval in range(96):
                prediction_rows.append({
                    'objective': objective,
                    'interval_start': date + pd.Timedelta(minutes=15 * interval),
                    'curve_kw': float(item['curve'][interval]),
                    'lgbm_q1_kw': float(item['q'][interval]),
                })

    outputs = {
        'comparison_summary': summaries,
        'accepted_moves': all_moves,
        'hourly_plans': schedules,
        'daily_production_comparison': daily_rows,
        'predictions': prediction_rows,
        'staff_wage_sensitivity': sensitivity,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig')
    (out / 'run_settings.json').write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    result = pd.DataFrame(summaries)
    plot_data = result.loc[result['basic_mode'].eq('monthly_peak_assumption')]
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(plot_data['objective'], plot_data['total_won_saving'])
    axes[0].set_title('기존 계획 대비 총비용 절감액(원)')
    axes[1].bar(plot_data['objective'], plot_data['peak_kw_saving'])
    axes[1].set_title('기존 계획 대비 월 최대전력 감소(kW)')
    for axis in axes:
        axis.axhline(0, color='black', linewidth=0.5)
        axis.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / 'cost_and_peak_comparison.png', dpi=160)
    plt.show()
    return result
