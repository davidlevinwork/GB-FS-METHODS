from kneed import KneeLocator

from ..config import config
from ..services import log_service
from ..models import OPERATION_MODE


def get_knees(results: dict):
    # REMOVE
    # del results['clustering'][0]
    # del results['clustering'][0]
    # for key in results['classification']:
    #    results['classification'][key] = results['classification'][key][2:]
    # REMOVE

    if config.operation_mode in [str(OPERATION_MODE.GBAFS), str(OPERATION_MODE.FULL_GBAFS)]:
        x = [res['k'] for res in results['clustering']]
        y = [res['silhouette']['MSS'] for res in results['clustering']]

        return get_knee(x=x, y=y, function="MSS"), results

    if config.operation_mode in [str(OPERATION_MODE.CS), str(OPERATION_MODE.FULL_CS)]:
        knee_results = {}
        heuristic_idx = get_heuristic_indexes(results=results)
        for sil_type in results['clustering'][0]['silhouette'].keys():
            if sil_type not in ['Silhouette', 'SS']:
                if sil_type == 'MSS':
                    x, y = get_x_y_values(results=results, function=f'Full {sil_type}', heuristic_idx=heuristic_idx)
                    knee_results.update(
                        get_knee(x=x, y=y, function=f'Full {sil_type}')
                    )
                x, y = get_x_y_values(results=results, function=sil_type, heuristic_idx=heuristic_idx)
                knee_results.update(
                    get_knee(x=x, y=y, function=sil_type)
                )
        return {
            'results': results,
            'knee_results': knee_results,
            'heuristic_idx': heuristic_idx
        }


def get_knee(x: list, y: list, function: str) -> dict:
    kn = KneeLocator(
        x,
        y,
        curve='concave',
        direction='increasing',
        interp_method='interp1d',
    )

    results = {
        'knee': kn.knee,
        'knee y': kn.knee_y,
        'knees': kn.all_knees,
        'knees y': kn.all_knees_y
    }

    log_service.log(f'[Knee Locator] : For {function} function, '
                    f'knee point founded in ({results["knee"]}, {results["knee y"]}).')
    return {function: results}


def get_x_y_values(results: dict, function: str, heuristic_idx: dict) -> tuple:
    if 'Naive' in function:
        x = [res['k'] for res in results['clustering']][:heuristic_idx['last_idx']]
        y = [x['silhouette'][function] for x in results['clustering']][:heuristic_idx['last_idx']]
        return x, y

    if function == 'MSS':
        x = [res['k'] for res in results['clustering']][:heuristic_idx['first_idx']]
        y = [x['silhouette'][function] for x in results['clustering']][:heuristic_idx['first_idx']]
        return x, y

    if function == 'Full MSS':
        x = [res['k'] for res in results['clustering']]
        y = [x['silhouette']['MSS'] for x in results['clustering']]
        return x, y

    if 'Greedy' in function:
        x = [res['k'] for res in results['clustering']][:heuristic_idx['last_idx']]
        y = [x['silhouette'][function] for x in results['clustering']][:heuristic_idx['last_idx']]
        return x, y


def get_heuristic_indexes(results: dict) -> dict:
    first_heuristic_idx, last_heuristic_idx = None, None

    for i, entry in enumerate([x['silhouette'] for x in results['clustering']]):
        greedy_key = next((key for key in entry.keys() if 'Greedy' in key), None)

        if greedy_key:
            if first_heuristic_idx is None and entry['MSS'] != entry[greedy_key]:
                first_heuristic_idx = i

            if last_heuristic_idx is None and entry[greedy_key] is None:
                last_heuristic_idx = i

        if first_heuristic_idx is not None and last_heuristic_idx is not None:
            break

    return {
        'first_idx': first_heuristic_idx,
        'last_idx': last_heuristic_idx
    }
