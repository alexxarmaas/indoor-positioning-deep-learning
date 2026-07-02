from flask import jsonify
from models.Capture import Capture,CaptureAux
from config import db

def buildCoordinateMapping(captures, pointsConfiguration):
    sourceCoordinates = []
    seenCoordinates = set()

    for capture in captures:
        coordinate = (capture.Position_x, capture.Position_y, capture.Position_z)
        if coordinate not in seenCoordinates:
            seenCoordinates.add(coordinate)
            sourceCoordinates.append(coordinate)

    targetCoordinates = [
        (float(point['X']), float(point['Y']), float(point['Z']))
        for point in pointsConfiguration.values()
    ]

    if len(sourceCoordinates) > len(targetCoordinates):
        raise ValueError(
            f'The database contains {len(sourceCoordinates)} points but the '
            f'configuration only defines {len(targetCoordinates)}'
        )

    return dict(zip(sourceCoordinates, targetCoordinates))


def createCapture(campaignId, pointsConfiguration):
    db.create_all()
    captureAux = CaptureAux.query.order_by(CaptureAux.Id.asc()).all()
    coordinateMapping = buildCoordinateMapping(captureAux, pointsConfiguration)
    captureIdMapping = {}

    for i in captureAux:
        normalizedCoordinate = coordinateMapping[(i.Position_x, i.Position_y, i.Position_z)]
        capture = Capture()
        capture.Id_campaign = campaignId
        capture.Date = i.Date
        capture.Light = i.Light
        capture.Temperature = i.Temperature
        capture.Relative_humidity = i.Relative_humidity
        capture.Absolute_humidity = i.Absolute_humidity
        capture.Position_x = normalizedCoordinate[0]
        capture.Position_y = normalizedCoordinate[1]
        capture.Position_z = normalizedCoordinate[2]
        capture.Platform_angle = i.Platform_angle
        capture.Dongle_rotation = i.Dongle_rotation
        db.session.add(capture)
        db.session.flush()
        captureIdMapping[i.Id] = capture.Id
    
    db.session.commit()
    db.session.remove()
    return captureIdMapping


def getCaptureIdsByCampaignId(campaignId):
    return [capture.Id for capture in Capture.query.filter_by(Id_campaign=campaignId).all()]

def getCapturesByCampaignId(campaignId, page=None, pageSize=None):
    query = Capture.query.filter_by(Id_campaign=campaignId)

    if page is None or pageSize is None:
        return jsonify([capture.serialize() for capture in query.all()])

    page = max(page, 0)
    pageSize = min(max(pageSize, 1), 100)
    total = query.count()
    captures = (
        query.order_by(Capture.Id.asc())
        .offset(page * pageSize)
        .limit(pageSize)
        .all()
    )
    return jsonify({
        'items': [capture.serialize() for capture in captures],
        'total': total,
        'page': page,
        'pageSize': pageSize
    })

def getCapturesById(id):
    return Capture.query.filter(
        Capture.Id.in_(id)
    ).all()

def deleteCaptureByCampaignId(campaignId):
    existingCapture = Capture().query.filter_by(Id_campaign=campaignId).delete()

    if existingCapture:
        db.session.commit()
