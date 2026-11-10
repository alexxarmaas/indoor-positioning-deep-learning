#!/usr/bin/env python3
"""Audita la alineación PA en IPSCampaign y la asignación espacial de las SQLite PR.
Solo lectura: nunca modifica bases de datos ni entrena modelos.
"""
import argparse
import configparser
import json
import math
import sqlite3
from pathlib import Path


def distance(a, b):
    return math.dist(tuple(map(float, a)), tuple(map(float, b)))


def xyz(points):
    return [(float(p['X']), float(p['Y']), float(p['Z'])) for p in points.values()]


def check_ips(name, campaign_id):
    from config import app
    from models.Campaign import Campaign
    from models.MethodPrediction import MethodPrediction
    from services.DataProcessService import getTFGCampaignMatrix

    with app.app_context():
        if campaign_id is not None:
            campaign = Campaign.query.filter_by(Id=campaign_id).first()
        else:
            matches = (Campaign.query.filter_by(Name=name)
                       .filter(Campaign.Id_related_campaign.isnot(None))
                       .order_by(Campaign.Id.desc()).all())
            if not matches:
                raise ValueError('No existe campaña PR con nombre ' + repr(name))
            if len(matches) > 1:
                print('AVISO: varias campañas con ese nombre; se usará la más reciente.')
            campaign = matches[0]

        if campaign is None or campaign.Id_related_campaign is None:
            raise ValueError('Campaña inexistente o sin PA relacionada')
        matrix, point_keys = getTFGCampaignMatrix(campaign.Id_related_campaign)
        backend = list(dict.fromkeys(tuple(map(float, p)) for p in point_keys))
        angular = xyz(json.loads(campaign.Ale_points))
        same_count = len(backend) == len(angular)
        same_order = same_count and all(distance(x, y) < 1e-6 for x, y in zip(backend, angular))
        print(f'Campaña: {campaign.Name}, ID={campaign.Id}; PA={campaign.Id_related_campaign}')
        print(f'Capturas PA válidas: {len(matrix)}')
        print(f'Puntos backend={len(backend)}, configuración Angular={len(angular)}')
        print(f'Orden PA idéntico: {"SÍ" if same_order else "NO"}')
        if not same_order:
            for i in range(max(len(backend), len(angular))):
                b = backend[i] if i < len(backend) else None
                a = angular[i] if i < len(angular) else None
                if b is None or a is None or distance(b, a) > 1e-6:
                    print(f'  índice={i} backend={b} Angular={a}')

        predictions = MethodPrediction.query.filter_by(Id_campaign=campaign.Id).all()
        for p in predictions:
            try:
                meta = json.loads(p.Method_description)
            except (TypeError, ValueError):
                continue
            if meta.get('method') not in ('MLP_TFG_ROBUST', 'MLP_TFG_MASK'):
                continue
            print(f'\nPredicción ID={p.Id}: {meta["method"]}')
            series = json.loads(p.Predicted_points)
            saved = json.loads(p.Mean_error)
            for key, values in series.items():
                if len(values) != len(backend):
                    print(f'  ERROR: {len(values)} predicciones, {len(backend)} posiciones reales')
                    continue
                be = [distance(pos, pred) for pos, pred in zip(backend, values)]
                print(f'  Media backend recalculada: {sum(be)/len(be):.6f} m')
                if same_count:
                    ae = [distance(pos, pred) for pos, pred in zip(angular, values)]
                    print(f'  Media Angular recalculada: {sum(ae)/len(ae):.6f} m')
                match = next((v for k, v in saved.items() if k.startswith(key + '[Mean=')), None)
                if match is None:
                    print('  AVISO: serie de errores no localizada')
                elif len(match) != len(be):
                    print('  ERROR: el número de errores guardados no coincide')
                else:
                    diff = max(abs(x-y) for x, y in zip(sorted(be), sorted(map(float, match))))
                    print(f'  Errores guardados: {"COINCIDEN" if diff < 1e-6 else "DIFIEREN"} (diferencia máxima={diff:.8f} m)')
        print('\nRESULTADO PA: ' + ('ALINEADA' if same_order else 'DESALINEADA: revisar orden de predicciones/configuración'))
        print('Esta prueba no certifica que las SQLite PR originales tengan el recorrido correcto.')
        return 0 if same_order else 2


def check_raw(conf, databases, tolerance):
    cp = configparser.ConfigParser()
    with open(conf, encoding='utf-8-sig') as f:
        cp.read_file(f)
    sections = sorted((s for s in cp.sections() if s.startswith('POINT_')),
                      key=lambda s: int(s.split('_')[-1]))
    grid = [(cp.getfloat(s, 'X'), cp.getfloat(s, 'Y'), cp.getfloat(s, 'Z')) for s in sections]
    failed = False
    print(f'Cuadrícula: {len(grid)} puntos; tolerancia={tolerance} m')
    for file in databases:
        uri = Path(file).resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True) as con:
            rows = con.execute('SELECT Position_x, Position_y, Position_z FROM Capture '
                               'GROUP BY Position_x, Position_y, Position_z ORDER BY MIN(Id)').fetchall()
        mismatches, out_of_tolerance, closest_indices = [], [], []
        for i, source in enumerate(rows):
            j, d = min(enumerate(distance(source, target) for target in grid), key=lambda item: item[1])
            closest_indices.append(j)
            if j != i:
                mismatches.append((i, j, round(d, 3)))
            if d > tolerance:
                out_of_tolerance.append((i, j, round(d, 3)))
        repeated = len(closest_indices) - len(set(closest_indices))
        ok = len(rows) == len(grid) and not mismatches and not out_of_tolerance and not repeated
        failed |= not ok
        print(f'\n{Path(file).name}: {len(rows)} puntos; '
              f'orden incorrecto={len(mismatches)}, fuera de tolerancia={len(out_of_tolerance)}, '
              f'asignaciones repetidas={repeated}: {"OK" if ok else "REVISAR"}')
        for i, j, d in mismatches[:12]:
            print(f'  Posición original {i} -> punto más cercano POINT_{j} (distancia={d} m)')
        for i, j, d in out_of_tolerance[:8]:
            print(f'  ALERTA punto {i}: distancia a POINT_{j}={d} m (supera tolerancia)')
    print('\nRESULTADO PR: ' + ('REVISAR asignación de coordenadas' if failed else 'RECORRIDOS COMPATIBLES'))
    return 2 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode', required=True)
    ip = sub.add_parser('ips', help='Verifica PA importada, predicciones y errores guardados')
    ip.add_argument('--campaign', default='varias')
    ip.add_argument('--campaign-id', type=int)
    raw = sub.add_parser('raw', help='Audita los recorridos espaciales de SQLite PR originales')
    raw.add_argument('--config', required=True)
    raw.add_argument('--databases', nargs='+', required=True)
    raw.add_argument('--tolerance', type=float, default=0.15)
    args = ap.parse_args()
    return check_ips(args.campaign, args.campaign_id) if args.mode == 'ips' else check_raw(args.config, args.databases, args.tolerance)


if __name__ == '__main__':
    raise SystemExit(main())
