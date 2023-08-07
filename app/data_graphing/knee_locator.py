from kneed import KneeLocator

from ..config import config
from ..services import log_service
from ..models import OPERATION_MODE


def get_knees(results: dict) -> dict:
    if config.operation_mode in [str(OPERATION_MODE.GBAFS), str(OPERATION_MODE.FULL_GBAFS)]:
        x = [res['k'] for res in results['clustering']]
        y = [res['silhouette']['MSS'] for res in results['clustering']]

        return get_knee(x=x, y=y, function="MSS")

    if config.operation_mode in [str(OPERATION_MODE.CS), str(OPERATION_MODE.FULL_CS)]:
        final_results = {}
        for sil_type in results['clustering'][0]['silhouette'].keys():
            if sil_type not in ['Silhouette', 'SS']:
                x, y = get_x_y_values(results=results, function=sil_type)
                final_results.update(
                    get_knee(x=x, y=y, function=sil_type)
                )
        return final_results


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


def get_x_y_values(results: dict, function: str) -> tuple:
    first_idx, last_idx = set_split_indexes(results=results)

    if function == 'Basic No Naive MSS':
        x = [res['k'] for res in results['clustering']][:last_idx]
        y = [x['silhouette'][function] for x in results['clustering']][:last_idx]
        return x, y

    if function == 'MSS':
        x = [res['k'] for res in results['clustering']][:first_idx]
        y = [x['silhouette'][function] for x in results['clustering']][:first_idx]
        return x, y

    if 'Greedy' in function:
        x = [res['k'] for res in results['clustering']][:last_idx]
        y = [x['silhouette']['MSS'] for x in results['clustering']][:first_idx] + \
            [x['silhouette'][function] for x in results['clustering']][first_idx:last_idx]
        return x, y


def set_split_indexes(results: dict) -> tuple:
    first_idx = last_idx = 0
    for sil_type in results['clustering'][0]['silhouette'].keys():
        if sil_type == 'Basic No Naive MSS':
            last_idx = [x['silhouette'][sil_type] for x in results['clustering']].index(0)
        if 'Greedy' in sil_type:
            first_idx = next(
                i for i, val in enumerate([res['silhouette'][sil_type] for res in results['clustering']]) if val != 0)

    return first_idx, last_idx
