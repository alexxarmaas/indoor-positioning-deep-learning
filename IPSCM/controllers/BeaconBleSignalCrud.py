from flask import jsonify
from controllers.CaptureCrud import getCaptureIdsByCampaignId
from models.BeaconBleSignal import BeaconBleSignal, BeaconBleSignalAux
from models.Capture import Capture
from config import db
from sqlalchemy import func


def normalizeMac(mac):
    """Return a stable representation for MAC addresses stored by campaigns."""
    if mac is None:
        return None

    normalized = mac.strip().lower().replace('-', ':')
    return normalized or None


def _normalizedMacColumn():
    return func.replace(func.lower(func.trim(BeaconBleSignal.Mac)), '-', ':')


def createBeaconBleSignal(captureIdMapping):
    db.create_all()
    beaconBleSignalAux = BeaconBleSignalAux.query.order_by(BeaconBleSignalAux.Id.asc()).yield_per(5000)
    signalBatch = []

    for i in beaconBleSignalAux:
        targetCaptureId = captureIdMapping.get(i.Id_capture)
        if targetCaptureId is None:
            raise ValueError(f'Signal {i.Id} references unknown capture {i.Id_capture}')

        signalBatch.append({
            'Id_capture': targetCaptureId,
            'N_reading': i.N_reading,
            'Date_hour': i.Date_hour,
            'Mac': normalizeMac(i.Mac),
            'Pack_size': i.Pack_size,
            'Channel': i.Channel,
            'RSSI': i.RSSI,
            'PDU_type': i.PDU_type,
            'CRC': i.CRC,
            'Protocol': i.Protocol,
            'Identificator': i.Identificator
        })

        if len(signalBatch) == 5000:
            db.session.bulk_insert_mappings(BeaconBleSignal, signalBatch)
            signalBatch.clear()

    if signalBatch:
        db.session.bulk_insert_mappings(BeaconBleSignal, signalBatch)
    
    db.session.commit()
    db.session.remove()


def getBeaconBleSignalByCampaignId(campaignId: int, page=None, pageSize=None):
    query = (
        BeaconBleSignal.query
        .join(Capture, BeaconBleSignal.Id_capture == Capture.Id)
        .filter(Capture.Id_campaign == campaignId)
    )

    if page is None or pageSize is None:
        return jsonify([signal.serialize() for signal in query.all()])

    page = max(page, 0)
    pageSize = min(max(pageSize, 1), 100)
    total = query.count()
    signals = (
        query.order_by(BeaconBleSignal.Id.asc())
        .offset(page * pageSize)
        .limit(pageSize)
        .all()
    )
    return jsonify({
        'items': [signal.serialize() for signal in signals],
        'total': total,
        'page': page,
        'pageSize': pageSize
    })

def getBeaconBleSignalIdentificatorByCampaignId(campaignId:int):
    captureIds = getCaptureIdsByCampaignId(campaignId)
    beaconBleSignals = BeaconBleSignal.query.filter(BeaconBleSignal.Id_capture.in_(captureIds)).all()
    return list(set(signal.Identificator for signal in beaconBleSignals))

def getBeaconBleSignalByIdChannelMacAndProtocol(id, channel, mac, protocol):
    return BeaconBleSignal.query.filter(
        BeaconBleSignal.Id_capture == id,
        BeaconBleSignal.Channel == channel,
        BeaconBleSignal.Mac.in_(mac),
        BeaconBleSignal.Protocol == protocol,
    ).all()

def getBeaconBleSignalById(id):
    return BeaconBleSignal.query.filter(
        BeaconBleSignal.Id_capture == id
    ).all()

def getBeaconBleSignalsByCaptureIds(captureIds):
    return BeaconBleSignal.query.filter(BeaconBleSignal.Id_capture.in_(captureIds)).all()


def countBeaconBleSignalsByCaptureIds(captureIds):
    return BeaconBleSignal.query.filter(BeaconBleSignal.Id_capture.in_(captureIds)).count()


def getMacsByCampaignId(campaignId):
    normalizedMac = _normalizedMacColumn()
    rows = db.session.query(normalizedMac).join(
        Capture,
        BeaconBleSignal.Id_capture == Capture.Id
    ).filter(
        Capture.Id_campaign == campaignId,
        normalizedMac.isnot(None),
        normalizedMac != ''
    ).distinct().all()

    return sorted(row[0] for row in rows)


def getObservedMacsByCampaignAndProtocol(campaignId, protocol):
    """Return the normalized MACs actually observed for a campaign/protocol."""
    normalizedMac = _normalizedMacColumn()
    normalizedProtocol = protocol.strip().lower()
    rows = db.session.query(normalizedMac).join(
        Capture,
        BeaconBleSignal.Id_capture == Capture.Id
    ).filter(
        Capture.Id_campaign == campaignId,
        func.lower(func.trim(BeaconBleSignal.Protocol)) == normalizedProtocol,
        normalizedMac.isnot(None),
        normalizedMac != ''
    ).distinct().all()

    return sorted(row[0] for row in rows)


def existsBeaconWithProtocol(campaignId, mac, protocol):
    captureIds = getCaptureIdsByCampaignId(campaignId)
    beaconBleSignal = BeaconBleSignal.query.filter(
        BeaconBleSignal.Id_capture.in_(captureIds),
        BeaconBleSignal.Mac == mac,
        BeaconBleSignal.Protocol == protocol
    ).first()
    
    if beaconBleSignal is not None:
        return True
    else:
        return False



def deleteBeaconBleSignalByCampaignId(campaignId:int):
    captureIds = getCaptureIdsByCampaignId(campaignId)
    BeaconBleSignal.query.filter(BeaconBleSignal.Id_capture.in_(captureIds)).delete()
    db.session.commit()


