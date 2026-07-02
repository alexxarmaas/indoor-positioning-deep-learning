import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpParams } from '@angular/common/http';
import { EMPTY, Observable, Subject, catchError, of, switchMap, takeUntil, throwError, timer } from 'rxjs';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private baseUrl = 'http://localhost:5000/api/';
  private destroy$ = new Subject();

  constructor(private http: HttpClient, private router: Router) {}

  getCampaigns(){
    return this.http.get(this.baseUrl + "campaigns");
  }

  getMethodPredictionsById(id: any){
    return this.http.get(this.baseUrl + `predictionsByCampaign/${id}`);
  }

  getTaskHistory(){
    return this.http.get(this.baseUrl + "taskHistory");
  }

  getImportProgress(){
    return this.http.get(this.baseUrl + "importProgress");
  }

  pollTaskHistory(): Observable<any> {
    return timer(0, 5000).pipe(
      takeUntil(this.destroy$),
      switchMap(() => {
        if (this.router.url === '/task-history') {
          return this.getTaskHistory();
        } else {
          this.destroy$.next(null);
          return EMPTY;
        }
      })
    );
  }

  getRelatedCampaignById(id: any){
    return this.http.get(this.baseUrl + `relatedCampaign/${id}`);
  }

  getCampaignImagesById(id: any): Observable<any> {
    return this.http.get(this.baseUrl + `campaignImages/${id}`, { responseType: 'blob' }).pipe(
      catchError((error: HttpErrorResponse) => {
        if (error.status === 404) {
          return of(null);
        }
        return throwError(() => error);
      })
    );
  }

  getDongleName(id: any){
    return this.http.get(this.baseUrl + `dongleReceptor/${id}`);
  }

  getCampaignSequence(id: any){
    return this.http.get(this.baseUrl + `campaignSequence/${id}`);
  }

  getCaptures(id: any, page?: number, pageSize?: number){
    return this.http.get(this.baseUrl + `capturesByCampaign/${id}`, {
      params: this.paginationParams(page, pageSize)
    });
  }

  getSignals(id: any, page?: number, pageSize?: number){
    return this.http.get(this.baseUrl + `signalsByCampaign/${id}`, {
      params: this.paginationParams(page, pageSize)
    });
  }

  private paginationParams(page?: number, pageSize?: number): HttpParams {
    let params = new HttpParams();
    if (page !== undefined && pageSize !== undefined) {
      params = params
        .set('page', page.toString())
        .set('pageSize', pageSize.toString());
    }
    return params;
  }

  getConfigs(id: any){
    return this.http.get(this.baseUrl + `configsByCampaign/${id}`);
  }

  applyMethod(formData: FormData): Observable<any> {
    return this.http.post(this.baseUrl + 'applyMethod', formData);
  }

  uploadCampaign(formData: FormData): Observable<any> {
    return this.http.post(this.baseUrl + 'uploadCampaign', formData);
  }

  deleteCampaignById(id: any){
    return this.http.delete(this.baseUrl + `deleteCampaign/${id}`);
  }
}
