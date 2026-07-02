from sqlalchemy import func, tuple_
from config import db

from models.MethodPredata import MethodPredata

def createMethodPredataBatch(batchData):
    db.create_all(bind_key=None)

    methodPredata_objects = [
        MethodPredata(
            Id_campaign=campaignId,
            Date=date,
            Channel=channel,
            Mac=mac,
            Dongle_rotation=rotation,
            Protocol=protocol,
            RSSI=rssi,
            Position_x=positionX,
            Position_y=positionY,
            Position_z=positionZ
        )
        for campaignId, date, channel, mac, rotation, protocol, rssi, positionX, positionY, positionZ in batchData
    ]

    db.session.bulk_save_objects(methodPredata_objects)
    db.session.commit()
    db.session.remove()


def getFilteredPredataSamples(campaignId, rssiSamples, dongle_rotation, mac, protocol, channel, positionX, positionY, positionZ):

    query = MethodPredata.query.order_by(MethodPredata.Date.asc(), MethodPredata.Id.asc())

    query = query.filter(MethodPredata.Id_campaign == campaignId)
    query = query.filter(MethodPredata.Dongle_rotation == dongle_rotation)
    query = query.filter(MethodPredata.Mac == mac)
    query = query.filter(MethodPredata.Protocol == protocol)
    query = query.filter(MethodPredata.Channel == channel)
    query = query.filter(func.round(MethodPredata.Position_x, 5) == round(positionX, 5))
    query = query.filter(func.round(MethodPredata.Position_y, 5) == round(positionY, 5))
    query = query.filter(func.round(MethodPredata.Position_z, 5) == round(positionZ, 5))

    if rssiSamples > 0:
        query = query.limit(rssiSamples)

    predata_samples = query.all()

    return predata_samples


def getUniqueCoordinatesByCampaignId(campaignId, limit):
    unique_coordinates = db.session.query(
        MethodPredata.Position_x,
        MethodPredata.Position_y,
        MethodPredata.Position_z,
        func.min(MethodPredata.Id).label('first_id')
    ).filter(
        MethodPredata.Id_campaign == campaignId
    ).group_by(
        MethodPredata.Position_x,
        MethodPredata.Position_y,
        MethodPredata.Position_z
    ).order_by(func.min(MethodPredata.Id)).limit(limit).all()

    unique_coordinates = [
        {'x': x, 'y': y, 'z': z}
        for x, y, z, _ in unique_coordinates
    ]

    return unique_coordinates


def deleteMethodPredataByCampaignId(campaignId):
    existingPredata = MethodPredata().query.filter_by(Id_campaign=campaignId).delete()

    if existingPredata:
        db.session.commit()
