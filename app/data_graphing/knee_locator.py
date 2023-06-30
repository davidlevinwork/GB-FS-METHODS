from kneed import KneeLocator


def get_knee(results: dict) -> dict:
    x = [res['k'] for res in results['clustering']]
    y = [res['silhouette']['MSS'] for res in results['clustering']]

    kn = KneeLocator(
        x,
        y,
        curve='concave',
        direction='increasing',
        interp_method='interp1d',
    )

    kn_res = {
        'knee': kn.knee,
        'knee y': kn.knee_y,
        'knees': kn.all_knees,
        'knees y': kn.all_knees_y
    }

    return kn_res
