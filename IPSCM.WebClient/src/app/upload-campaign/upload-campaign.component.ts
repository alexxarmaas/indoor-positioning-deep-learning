import { Component, OnDestroy } from '@angular/core';
import { ToastrService } from 'ngx-toastr';
import { FormGroup, FormBuilder, Validators } from '@angular/forms';
import { ApiService } from '../services/api.service';
import { Router } from '@angular/router';
import { Subscription, catchError, finalize, of, switchMap, tap, timer } from 'rxjs';



@Component({
  selector: 'app-upload-campaign',
  templateUrl: './upload-campaign.component.html',
  styleUrls: ['./upload-campaign.component.sass']
})
export class UploadCampaignComponent implements OnDestroy {

  uploadForm: FormGroup;

  constructor(
    private toastr: ToastrService, 
    private fb: FormBuilder, 
    private apiService: ApiService,
    private router: Router)
  {
    this.uploadForm = this.fb.group({
      name: ['', [Validators.required]],
      date: [new Date().toISOString().substring(0, 10), Validators.required],
      description: []
    });
  }
  
  selectedAleatoryFile: File | null = null;
  selectedReferenceFiles: File[] = [];
  selectedConfFiles: File[] = [];
  selectedImages: File[] = [];
  aleatoryFileUploaded = false;
  referenceFilesUploaded = false;
  imageUploaded = false;
  confUploaded = false;
  allConfsUploaded = false;
  aleatoryFileName = '';
  referenceFileNames: string[] = [];
  imageNames: string[] = [];
  confNames: string[] = [];
  isDraggingAleatoryFile = false;
  isDraggingReferenceFiles = false;
  isDraggingImage = false;
  isDraggingConf = false;
  isUploading = false;
  importProgress = 0;
  importStatusMessage = '';
  private importProgressSubscription?: Subscription;
  fileAllowedExtensions = /(\.sqlite3)$/i;
  imageAllowedExtensions = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
  confAllowedExtensions = /(\.conf)$/i;

  onDragOver(event: any, type: string){
    event.preventDefault();
    event.stopPropagation();

    if(type == 'image'){
      this.isDraggingImage = true;
    }
    else if(type == 'aleatoryFile') this.isDraggingAleatoryFile = true;
    else if(type == 'referenceFiles') this.isDraggingReferenceFiles = true;
    else this.isDraggingConf = true;
  }

  onDragLeave(event: any, type: string){
    event.preventDefault();
    event.stopPropagation();
    
    if(type == 'image'){
      this.isDraggingImage = false;
    }
    else if(type == 'aleatoryFile') this.isDraggingAleatoryFile = false;
    else if(type == 'referenceFiles') this.isDraggingReferenceFiles = false;
    else this.isDraggingConf = false;
  }

  onSelected(event: any, type: string) {
    this.onSelectedOrOnDrop(event.target.files, type);
  }

  onDrop(event: any, type: string) {
    event.preventDefault();
    this.onSelectedOrOnDrop(event.dataTransfer.files, type);
  }
  
  onSelectedOrOnDrop(files: any, type: string) {
    if(type == 'image'){
      this.isDraggingImage = false;
      for(let i = 0; i < files.length; i++){
        var file = files[i];
     
        if (!this.imageAllowedExtensions.includes(file.type)) {
          this.toastr.error('El archivo no posee un formato válido (JPEG, JPG, PNG, GIF)', 'Error de formato');
        }
        else{
          this.imageUploaded = true;
          this.selectedImages?.push(file);
          this.imageNames?.push(file.name);
        }
      }
    } else if(type == 'aleatoryFile'){
      this.isDraggingAleatoryFile = false;
      const newFiles = Array.from(files) as File[];

      if (newFiles.length !== 1 || !this.fileAllowedExtensions.test(newFiles[0].name)) {
        this.toastr.error('El archivo no posee un formato válido (SQLITE3)', 'Error de formato');
        return;
      }

      this.selectedAleatoryFile = newFiles[0];
      this.aleatoryFileName = newFiles[0].name;
      this.aleatoryFileUploaded = true;
      return;
    } else if(type == 'referenceFiles'){
      this.isDraggingReferenceFiles = false;
      const newFiles = Array.from(files) as File[];

      if (newFiles.some(file => !this.fileAllowedExtensions.test(file.name))) {
        this.toastr.error('El archivo no posee un formato válido (SQLITE3)', 'Error de formato');
        return;
      }

      const selectedFiles = new Set(
        this.selectedReferenceFiles.map(file => `${file.name}:${file.size}:${file.lastModified}`)
      );

      for (const file of newFiles) {
        const fileKey = `${file.name}:${file.size}:${file.lastModified}`;
        if (!selectedFiles.has(fileKey)) {
          this.selectedReferenceFiles.push(file);
          selectedFiles.add(fileKey);
        }
      }

      this.referenceFilesUploaded = this.selectedReferenceFiles.length > 0;
      this.referenceFileNames = this.selectedReferenceFiles.map((file, index) =>
        `Referencia ${index + 1}: ${file.name}`
      );
      return;
    }
    else{
      this.isDraggingConf = false;
      if (files.length > 3 || this.selectedConfFiles.length == 3) {
        this.toastr.error('Solo se permiten cargar tres ficheros de forma simultánea', 'Limite de ficheros excedido');
        console.log(this.allConfsUploaded);
        console.log(this.confUploaded);
        return
      }

        for(let i = 0; i < files.length; i++){
          var checkFile = files[i];

          if (checkFile && !this.confAllowedExtensions.exec(checkFile.name)) {
            this.toastr.error('El archivo no posee un formato válido (CONF)', 'Error de formato');
            return
          } 
          else {
              this.confUploaded = true;
              this.selectedConfFiles?.push(checkFile);
              this.confNames?.push(checkFile.name);
              if (this.selectedConfFiles.length == 3) {
                this.allConfsUploaded = true;
              }

          }
        }
    }
  }

  clearAleatoryFile(): void {
    this.selectedAleatoryFile = null;
    this.aleatoryFileName = '';
    this.aleatoryFileUploaded = false;
  }

  clearReferenceFiles(): void {
    this.selectedReferenceFiles = [];
    this.referenceFileNames = [];
    this.referenceFilesUploaded = false;
  }

  uploadCampaign(): void {
    if (this.isUploading) {
      return;
    }

    const formData = new FormData();

    formData.append('name', this.uploadForm.get('name')?.value);
    formData.append('date', this.uploadForm.get('date')?.value);
    formData.append('description', this.uploadForm.get('description')?.value);

    if (this.selectedImages) {
      for (let i = 0; i < this.selectedImages.length; i++) {
        formData.append('images', this.selectedImages[i]);
      }
    }
    
    if (this.selectedAleatoryFile) {
      formData.append('aleatoryFile', this.selectedAleatoryFile);
    }

    for (const referenceFile of this.selectedReferenceFiles) {
      formData.append('referenceFiles', referenceFile);
    }

    if (this.selectedConfFiles){
      for (let i = 0; i < this.selectedConfFiles.length; i++) {
        formData.append('confs', this.selectedConfFiles[i]);
      }
    }
    this.isUploading = true;
    this.importProgress = 0;
    this.importStatusMessage = 'Subiendo archivos al servidor...';
    this.startImportProgressPolling();
    this.toastr.warning('La importación puede tardar varios minutos.', 'Operación en curso', { disableTimeOut: true });

    this.apiService.uploadCampaign(formData).pipe(
      tap(response => {
        this.importProgress = 100;
        this.importStatusMessage = 'Campaña importada correctamente';
        this.toastr.clear();
        this.toastr.success('¡La campaña se ha guardado correctamente!', 'Operación completada', { timeOut: 3000 });
        this.router.navigateByUrl('/', { skipLocationChange: true }).then(() => {
          this.router.navigate(['/campaigns']);
        });
      }),
      catchError(error => {
        this.importStatusMessage = error.status === 409
          ? 'Ya hay otra importación en curso'
          : 'Ocurrió un error durante la importación';
        this.toastr.clear();
        this.toastr.error(this.importStatusMessage, 'Operación no completada', { timeOut: 5000 });
        return of(null);
      }),
      finalize(() => {
        this.isUploading = false;
        this.importProgressSubscription?.unsubscribe();
      })
    ).subscribe();
  }

  private startImportProgressPolling(): void {
    this.importProgressSubscription?.unsubscribe();
    this.importProgressSubscription = timer(0, 750).pipe(
      switchMap(() => this.apiService.getImportProgress())
    ).subscribe({
      next: (progress: any) => {
        if (progress.state === 'running') {
          this.importProgress = progress.percentage ?? this.importProgress;
          this.importStatusMessage = progress.message ?? this.importStatusMessage;
        }
      },
      error: () => {}
    });
  }

  ngOnDestroy(): void {
    this.importProgressSubscription?.unsubscribe();
  }
}
