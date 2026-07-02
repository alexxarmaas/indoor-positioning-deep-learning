from copy import deepcopy
from threading import Lock


_importLock = Lock()
_statusLock = Lock()
_importStatus = {
    'state': 'idle',
    'stage': 'idle',
    'message': 'No hay ninguna importacion en curso',
    'percentage': 0,
    'current': 0,
    'total': 0
}


def startImport(totalDatabases):
    if not _importLock.acquire(blocking=False):
        return False

    updateImportProgress(
        state='running',
        stage='upload',
        message='Recibiendo archivos...',
        percentage=0,
        current=0,
        total=totalDatabases
    )
    return True


def updateImportProgress(**changes):
    with _statusLock:
        _importStatus.update(changes)


def completeImport(message='Campana importada correctamente'):
    updateImportProgress(
        state='completed',
        stage='completed',
        message=message,
        percentage=100
    )


def failImport(message):
    updateImportProgress(
        state='error',
        stage='error',
        message=message
    )


def releaseImport():
    if _importLock.locked():
        _importLock.release()


def getImportProgress():
    with _statusLock:
        return deepcopy(_importStatus)
