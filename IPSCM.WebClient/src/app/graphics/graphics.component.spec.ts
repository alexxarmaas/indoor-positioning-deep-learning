import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { FormBuilder } from '@angular/forms';
import { ToastrService } from 'ngx-toastr';
import { of } from 'rxjs';

import { GraphicsComponent } from './graphics.component';
import { ApiService } from '../services/api.service';

describe('GraphicsComponent', () => {
  let component: GraphicsComponent;
  let fixture: ComponentFixture<GraphicsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GraphicsComponent ],
      providers: [
        FormBuilder,
        {
          provide: ActivatedRoute,
          useValue: { queryParams: of({}) }
        },
        {
          provide: ApiService,
          useValue: { getMethodPredictionsById: () => of(0) }
        },
        {
          provide: ToastrService,
          useValue: { warning: () => undefined }
        }
      ],
      schemas: [NO_ERRORS_SCHEMA]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GraphicsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should summarize positioning errors in metres', () => {
    const summary = component.summarizeErrors([1, 2, 3, 4]);

    expect(summary.count).toBe(4);
    expect(summary.mean).toBe(2.5);
    expect(summary.median).toBe(2);
    expect(summary.rmse).toBeCloseTo(Math.sqrt(7.5), 8);
    expect(summary.p90).toBe(4);
    expect(summary.maximum).toBe(4);
  });

  it('should hide the beacon inventory from visible series labels', () => {
    const description = JSON.stringify({
      beaconCount: 2,
      beaconMacs: ['aa:aa', 'bb:bb']
    });
    const storedLabel =
      "[WKNN][iBeacon][37][20][auto][manhattan][2]" +
      "[['aa:aa', 'bb:bb']][k=3]";

    const visibleLabel = component.getDisplaySeriesLabel(
      storedLabel, description
    );

    expect(visibleLabel).toBe(
      '[WKNN][iBeacon][37][20][auto][manhattan][k=3]'
    );
  });
});
