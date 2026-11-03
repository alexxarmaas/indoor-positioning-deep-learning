from controllers.CaptureCrud import (
    getCaptureIdsByCampaignId,
    getCapturesById
)

from controllers.BeaconBleSignalCrud import (
    getBeaconBleSignalsByCaptureIds
)

from services.TFGFeaturesService import build_tfg_dataset

def load_tfg_campaign_dataset(campaign_id):

    capture_ids = getCaptureIdsByCampaignId(campaign_id)

    if not capture_ids:
        raise ValueError(f"La campaña {campaign_id} no contiene capturas")

    captures = getCapturesById(capture_ids)
    signals = getBeaconBleSignalsByCaptureIds(capture_ids)

    X, y, ids = build_tfg_dataset(captures, signals)

    return X, y, ids