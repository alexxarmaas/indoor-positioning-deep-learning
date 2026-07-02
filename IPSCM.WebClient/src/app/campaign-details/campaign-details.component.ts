import { Component, ViewChild } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { ApiService } from '../services/api.service';
import JSZip from 'jszip';
import { OwlOptions } from 'ngx-owl-carousel-o';
import { Observable, forkJoin, map, of, switchMap } from 'rxjs';
import { MatButtonToggleChange } from '@angular/material/button-toggle';
import { MatTableDataSource } from '@angular/material/table';
import { MatPaginator, PageEvent } from '@angular/material/paginator';

interface PagedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
}

@Component({
  selector: 'app-campaign-details',
  templateUrl: './campaign-details.component.html',
  styleUrls: ['./campaign-details.component.sass'],
  providers: [],
})
export class CampaignDetailsComponent {

  campaignImages: string[] = [];
  campaign: any;
  campaignDetails: any;
  relatedCampaignDetails: any;
  selectedCampaign: any;
  selectedCampaignType: 'aleatory' | 'reference' = 'aleatory';
  campaignConfigs = 0;
  opt = 1;
  loading = true;
  capturesLoading = false;
  signalsLoading = false;

  defaultPageSize = 10;
  pageSizes = [10, 25, 50, 100];
  capturesDataSource = new MatTableDataSource<any>([]);
  signalsDataSource = new MatTableDataSource<any>([]);
  configsDataSource = new MatTableDataSource<any>([]);
  capturesPageIndex = 0;
  signalsPageIndex = 0;
  capturesPageSize = this.defaultPageSize;
  signalsPageSize = this.defaultPageSize;
  capturesTotal = 0;
  signalsTotal = 0;

  private capturesLoadedCampaignId?: number;
  private signalsLoadedCampaignId?: number;

  capturesColumns = ['Id', 'Date', 'Light', 'Temperature', 'Relative_humidity', 'Absolute_humidity', 'Position_x', 'Position_y', 'Position_z', 'Platform_angle', 'Dongle_rotation'];
  signalsColumns = ['Id', 'N_reading', 'Date_hour', 'Mac', 'Pack_size', 'Channel', 'RSSI', 'PDU_type', 'CRC', 'Protocol', 'Identificator'];
  configsColumns = ['Id', 'Mac', 'Power', 'Calibrated_power', 'Update_frequency', 'Identificator'];

  @ViewChild('configsPaginator') configsPaginator?: MatPaginator;

  customOptions: OwlOptions = {
    loop: true,
    smartSpeed: 1000,
    autoplayTimeout: 6000,
    autoplay: true,
    mouseDrag: true,
    touchDrag: false,
    pullDrag: false,
    dots: true,
    navText: ['', ''],
    responsive: {
      0: { items: 1 },
      400: { items: 2 },
      740: { items: 3 }
    },
    nav: false,
  };

  constructor(private route: ActivatedRoute, private apiService: ApiService) {
    this.route.queryParams.subscribe(params => {
      if (params['data']) {
        this.campaign = JSON.parse(params['data']);
      }
    });

    if (!this.campaign) {
      this.loading = false;
      return;
    }

    this.loadCampaignImages();

    forkJoin([
      this.campaignData(),
      this.relatedCampaignData()
    ]).subscribe({
      next: ([referenceCampaign, aleatoryCampaign]) => {
        this.campaignDetails = referenceCampaign;
        this.relatedCampaignDetails = aleatoryCampaign;

        if (aleatoryCampaign) {
          this.selectCampaign(aleatoryCampaign, 'aleatory');
        } else {
          this.selectCampaign(referenceCampaign, 'reference');
        }
        this.loading = false;
      },
      error: () => {
        this.loading = false;
      }
    });
  }

  ngAfterViewInit(): void {
    if (this.configsPaginator) {
      this.configsDataSource.paginator = this.configsPaginator;
    }
  }

  campaignData(): Observable<any> {
    return this.loadCampaignData(this.campaign);
  }

  relatedCampaignData(): Observable<any> {
    if (!this.campaign.Id_related_campaign) {
      return of(null);
    }

    return this.apiService.getRelatedCampaignById(this.campaign.Id_related_campaign).pipe(
      switchMap((relatedCampaign: any) => this.loadCampaignData(relatedCampaign))
    );
  }

  private loadCampaignData(campaign: any): Observable<any> {
    return forkJoin([
      this.apiService.getDongleName(campaign.Id_dongle),
      this.apiService.getCampaignSequence(campaign.Id_campaign_sequence),
      this.apiService.getConfigs(campaign.Id)
    ]).pipe(
      map(([dongleName, campaignSequence, configs]) => ({
        ...campaign,
        dongleName,
        campaignSequence,
        configs
      }))
    );
  }

  switchCampaignData(event: MatButtonToggleChange): void {
    const type = event.value as 'aleatory' | 'reference';
    const details = type === 'aleatory'
      ? this.relatedCampaignDetails
      : this.campaignDetails;

    if (details) {
      this.selectCampaign(details, type);
    }
  }

  private selectCampaign(details: any, type: 'aleatory' | 'reference'): void {
    this.selectedCampaign = details;
    this.selectedCampaignType = type;
    this.configsDataSource.data = details.configs ?? [];
    this.campaignConfigs = this.configsDataSource.data.length;
    this.configsPaginator?.firstPage();

    this.capturesPageIndex = 0;
    this.signalsPageIndex = 0;
    this.capturesTotal = details.campaignSequence?.Capture ?? 0;
    this.signalsTotal = details.campaignSequence?.Beacon_BLE_Signal ?? 0;
    this.capturesDataSource.data = [];
    this.signalsDataSource.data = [];
    this.capturesLoadedCampaignId = undefined;
    this.signalsLoadedCampaignId = undefined;

    if (this.opt === 2) {
      this.loadSignals();
    } else if (this.opt === 1) {
      this.loadCaptures();
    }
  }

  onCapturesPage(event: PageEvent): void {
    this.capturesPageIndex = event.pageIndex;
    this.capturesPageSize = event.pageSize;
    this.loadCaptures();
  }

  onSignalsPage(event: PageEvent): void {
    this.signalsPageIndex = event.pageIndex;
    this.signalsPageSize = event.pageSize;
    this.loadSignals();
  }

  private loadCaptures(): void {
    const campaignId = this.selectedCampaign?.Id;
    if (!campaignId) {
      return;
    }

    this.capturesLoading = true;
    this.apiService.getCaptures(
      campaignId,
      this.capturesPageIndex,
      this.capturesPageSize
    ).subscribe({
      next: (response: any) => {
        if (this.selectedCampaign?.Id !== campaignId) {
          return;
        }
        const page = response as PagedResponse<any>;
        this.capturesDataSource.data = page.items;
        this.capturesTotal = page.total;
        this.capturesLoadedCampaignId = campaignId;
        this.capturesLoading = false;
      },
      error: () => {
        this.capturesLoading = false;
      }
    });
  }

  private loadSignals(): void {
    const campaignId = this.selectedCampaign?.Id;
    if (!campaignId) {
      return;
    }

    this.signalsLoading = true;
    this.apiService.getSignals(
      campaignId,
      this.signalsPageIndex,
      this.signalsPageSize
    ).subscribe({
      next: (response: any) => {
        if (this.selectedCampaign?.Id !== campaignId) {
          return;
        }
        const page = response as PagedResponse<any>;
        this.signalsDataSource.data = page.items;
        this.signalsTotal = page.total;
        this.signalsLoadedCampaignId = campaignId;
        this.signalsLoading = false;
      },
      error: () => {
        this.signalsLoading = false;
      }
    });
  }

  optionSelected(option: string): void {
    switch (option) {
      case 'captures':
        this.opt = 1;
        if (this.capturesLoadedCampaignId !== this.selectedCampaign?.Id) {
          this.loadCaptures();
        }
        break;
      case 'beacon_signal':
        this.opt = 2;
        if (this.signalsLoadedCampaignId !== this.selectedCampaign?.Id) {
          this.loadSignals();
        }
        break;
      case 'beacon_config':
        this.opt = 3;
        break;
    }
  }

  private loadCampaignImages(): void {
    this.apiService.getCampaignImagesById(this.campaign.Id).subscribe((response: Blob) => {
      if (response != null) {
        const zipImages = new JSZip();
        zipImages.loadAsync(response).then(images => {
          let count = 0;
          Object.keys(images.files).forEach(filename => {
            images.files[filename].async('base64').then(imageData => {
              const imageUrl = 'data:image/jpeg;base64,' + imageData;
              if (count < 3 && Object.keys(images.files).length === 1) {
                for (let i = 0; i < 2; i++) {
                  this.campaignImages.push(imageUrl);
                  count++;
                }
              } else {
                this.campaignImages.push(imageUrl);
                count++;
              }
            });
          });
        });
      } else {
        const defaultImageUrl = 'assets/img/defaultCampaignImage.png';
        this.campaignImages.push(defaultImageUrl);
        this.campaignImages.push(defaultImageUrl);
      }
    });
  }
}