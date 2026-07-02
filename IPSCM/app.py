from config import app, db
from flask_restful import Api
from flask_cors import CORS
from sqlalchemy import text
from routes import *


with app.app_context():
    db.create_all(bind_key=None)
    db.session.execute(text(
        'CREATE INDEX IF NOT EXISTS ix_capture_campaign '
        'ON "Capture" ("Id_campaign")'
    ))
    db.session.execute(text(
        'CREATE INDEX IF NOT EXISTS ix_signal_capture '
        'ON "Beacon_BLE_Signal" ("Id_capture")'
    ))
    db.session.commit()

    api = Api(app)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    #get resources
    api.add_resource(ServerStatus, '/')
    api.add_resource(GetCampaigns, '/api/campaigns')
    api.add_resource(GetImportProgress, '/api/importProgress')
    api.add_resource(GetTaskHistory, '/api/taskHistory')
    api.add_resource(GetRelatedCampaignById, '/api/relatedCampaign/<int:relatedCampaignId>')
    api.add_resource(GetCampaignImagesById, '/api/campaignImages/<int:campaignId>')
    api.add_resource(GetDongleReceptor, '/api/dongleReceptor/<int:dongleId>')
    api.add_resource(GetCampaignSequence, '/api/campaignSequence/<int:sequenceId>')
    api.add_resource(GetCapturesByCampaignId, '/api/capturesByCampaign/<int:campaignId>')
    api.add_resource(GetPredictionsByCampaignId, '/api/predictionsByCampaign/<int:campaignId>')
    api.add_resource(GetBeaconBleSignalByCampaignId, '/api/signalsByCampaign/<int:campaignId>')
    api.add_resource(GetBeaconConfigurationByCampaignId, '/api/configsByCampaign/<int:campaignId>')

    #post resources
    api.add_resource(UploadCampaign, '/api/uploadCampaign')
    api.add_resource(DataProcessing, '/api/applyMethod')

    #delete resources
    api.add_resource(DeteteCampaignById, '/api/deleteCampaign/<int:campaignId>')
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
