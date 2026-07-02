import { Component, ElementRef, ViewChild } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Chart, ChartConfiguration, ChartType } from 'chart.js';
import { ApiService } from '../services/api.service';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import annotationPlugin from 'chartjs-plugin-annotation';
import { ToastrService } from 'ngx-toastr';


@Component({
  selector: 'app-graphics',
  templateUrl: './graphics.component.html',
  styleUrls: ['./graphics.component.sass']
})
export class GraphicsComponent {

  @ViewChild('chartCanvas', { static: true }) chartCanvas!: ElementRef<HTMLCanvasElement>;
  campaignId: any;
  campaignName?: string;
  alePoints?: string;
  campaignMethodPrediction: any = [];
  filteredData: any = [];
  totalSeries: any = [];
  selectedSeries: any = [];
  precisionSummary: any[] = [];
  accuracySummary: any[] = [];
  campaignFilteredPoints: any = [];
  campaoignFilteredErrors: any = [];
  methods: string[] = ["WKNN", "DeepLearning", "SVR", "NuSVR", "LinearSVR"];
  protocols: string[] = ["Eddystone", "iBeacon"];
  channels: string[] = ["37", "38", "39"];
  selectAllProtocolsCheck: boolean = false;
  selectAllChannelsCheck: boolean = false;
  selectAllMetricsCheck: boolean = false;
  selectAllAlgorithmsCheck: boolean = false;

  metricParams: string[] = ["manhattan", "chebyshev", "minkowski"]
  algorithmParams: string[] = ["brute", "ball_tree", "kd_tree", "auto"]
  kernelParams: string[] = ["linear", "poly", "sigmoid", "rbf"]
  gammaParams: string[] = ["auto", "scale", "range"]
  WKNNparams: boolean = false;
  SVRparams: boolean = false;
  NuSVRparams: boolean = false;
  LinearSVRparams: boolean = false;
  SVRgammaRange: boolean = false;
  NuSVRgammaRange: boolean = false;

  selectedMethod: any;
  form!: FormGroup;

  ngOnInit(){
    this.methods.splice(2, 0, 'DeepLearningPyTorch', 'DeepLearningHybrid');
    this.form = this.formBuilder.group({
      methods: ['', Validators.required],
      protocols: ['', Validators.required],
      channels: ['', Validators.required],
      metrics: ['', Validators.required],
      algorithms: ['', Validators.required],
      kernels: ['', Validators.required],
      gamma: ['', Validators.required],
      kRangeStart: [1],
      kRangeEnd: [20],
      cRangeStart: [0.1],
      cRangeEnd: [20],
      gRangeStart: [0.01],
      gRangeEnd: [1],
      nuRangeStart: [0.01],
      nuRangeEnd: [1],
      iRangeStart: [0],
      iRangeEnd: [10000]
    });

    Chart.register(annotationPlugin);
  }

  
  constructor(private route: ActivatedRoute, private apiService: ApiService, private formBuilder: FormBuilder, private toastr: ToastrService){
    this.route.queryParams.subscribe(params => {
      if(params['id']){
        this.campaignId = params['id'];
      }
      if(params['name']){
        this.campaignName = params['name'];
        if(this.lineChartOptions){
          this.lineChartOptions.plugins = {
            ...this.lineChartOptions.plugins,
            title: {
              display: true,
              text: this.campaignName + " - Gráfica de precisión",
              font:{
                size:24
              }
            }
          };
        }

        if(this.pointsChartOptions){
          this.pointsChartOptions.plugins = {
            ...this.pointsChartOptions.plugins,
            title: {
              display: true,
              text: this.campaignName + " - Gráfica de exactitud",
              font:{
                size:24
              }
            }
          };
        }

        if(params['alePoints']){
          this.alePoints = JSON.parse(params['alePoints']);
        }
      }
    })

    this.apiService.getMethodPredictionsById(this.campaignId).subscribe(res=> {
      
      if(res != 0){
        this.campaignMethodPrediction = res
      }
    })
  }
  
  selectAllProtocols(){
    const protocolsControl = this.form.get('protocols');
    const protocolsToggle = this.form.get('protocols')?.value;

    const allProtocols = this.protocols.slice();
    
    if(protocolsControl){
      if (protocolsToggle.length == 1) {
        protocolsControl.setValue(allProtocols);
        this.selectAllProtocolsCheck = true;
      } else {
        protocolsControl.setValue([]);
        this.selectAllProtocolsCheck = false;
      }
    }
  }

  selectAllChannels(){
    const channelsControl = this.form.get('channels');
    const channelsToggle = this.form.get('channels')?.value;

    const allChannels = this.channels.slice();
    
    if(channelsControl){
      if (channelsToggle.length == 1) {
        channelsControl.setValue(allChannels);
        this.selectAllChannelsCheck = true;
      } else {
        channelsControl.setValue([]);
        this.selectAllChannelsCheck = false;
      }
    }
  }

  selectAllMetrics(){
    const metricsControl = this.form.get('metrics');
    const metricsToggle = this.form.get('metrics')?.value;

    const allMetrics = this.metricParams.slice();
    
    if(metricsControl){
      if (metricsToggle.length == 1) {
        metricsControl.setValue(allMetrics);
        this.selectAllMetricsCheck = true;
      } else {
        metricsControl.setValue([]);
        this.selectAllMetricsCheck = false;
      }
    }
  }

  selectAllAlgorithms(){
    const algorithmsControl = this.form.get('algorithms');
    const algorithmsToggle = this.form.get('algorithms')?.value;

    const allAlgorithms = this.algorithmParams.slice();
    
    if(algorithmsControl){
      if (algorithmsToggle.length == 1) {
        algorithmsControl.setValue(allAlgorithms);
        this.selectAllAlgorithmsCheck = true;
      } else {
        algorithmsControl.setValue([]);
        this.selectAllAlgorithmsCheck = false;
      }
    }
  }

  selectAllKernels(){
    const kernelsControl = this.form.get('kernels');
    const kernelsToggle = this.form.get('kernels')?.value;

    const allKernels = this.kernelParams.slice();
    
    if(kernelsControl){
      if (kernelsToggle.length == 1) {
        kernelsControl.setValue(allKernels);
        this.selectAllAlgorithmsCheck = true;
      } else {
        kernelsControl.setValue([]);
        this.selectAllAlgorithmsCheck = false;
      }
    }
  }

  onMethodSelected(method: string){
    this.selectedMethod = method;
    this.validateAndClearFields();

    this.WKNNparams = false;
    this.SVRparams = false;
    this.NuSVRparams = false;
    this.LinearSVRparams = false;

    switch(this.selectedMethod){
      case "WKNN":
        this.WKNNparams = true;
        this.form.get('gamma')?.setValidators(null);
        this.form.get('kernels')?.setValidators(null);
        break;
      case "DeepLearning":
      case 'DeepLearningPyTorch':
      case 'DeepLearningHybrid':
        if (this.selectedMethod === 'DeepLearningHybrid') {
          this.form.get('channels')?.setValidators(null);
        }
        this.form.get('algorithms')?.setValidators(null);
        this.form.get('metrics')?.setValidators(null);
        this.form.get('gamma')?.setValidators(null);
        this.form.get('kernels')?.setValidators(null);
        break;
      case "SVR":
        this.SVRparams = true;
        this.form.get('algorithms')?.setValidators(null);
        this.form.get('metrics')?.setValidators(null);
        break;
      case "NuSVR":
        this.NuSVRparams = true;
        this.form.get('algorithms')?.setValidators(null);
        this.form.get('metrics')?.setValidators(null);
        break;
      case "LinearSVR":
        this.LinearSVRparams = true;
        this.form.get('algorithms')?.setValidators(null);
        this.form.get('metrics')?.setValidators(null);
        this.form.get('gamma')?.setValidators(null);
        this.form.get('kernels')?.setValidators(null);
        break;
    }

    Object.keys(this.form?.controls).forEach(key => {
      this.form?.get(key)?.updateValueAndValidity();
    });
  }

  validateAndClearFields(){
    this.form.get('methods')?.setValidators(Validators.required);
    this.form.get('protocols')?.setValidators(Validators.required);
    this.form.get('channels')?.setValidators(Validators.required);
    this.form.get('metrics')?.setValidators(Validators.required);
    this.form.get('metrics')?.reset();
    this.form.get('algorithms')?.setValidators(Validators.required);
    this.form.get('algorithms')?.reset();
    this.form.get('gamma')?.setValidators(Validators.required);
    this.form.get('gamma')?.reset();
    this.form.get('kernels')?.setValidators(Validators.required);
    this.form.get('kernels')?.reset();
  }

  onSVRGammaSelected(gamma: string){
    this.SVRgammaRange = false;

    if(gamma === "range"){
      this.SVRgammaRange = true;
    }
  }

  onNuSVRgammaSelected(gamma: string){
      this.NuSVRgammaRange = false;

      if(gamma === "range"){
        this.NuSVRgammaRange = true;
      }
  }

  onFilterSelected(chartTipe:string){
    this.filteredData = [...this.campaignMethodPrediction];

    this.filteredData = this.filterData(this.form.get('methods')?.value, "method", this.filteredData);
    this.filteredData = this.filterData(this.form.get('protocols')?.value, "protocol", this.filteredData);
    if (this.selectedMethod !== 'DeepLearningHybrid') {
      this.filteredData = this.filterData(this.form.get('channels')?.value, "channel", this.filteredData);
    }
    
    switch(this.selectedMethod){
      case "WKNN":
        this.filteredData = this.filterData(this.form.get('metrics')?.value, "metric", this.filteredData);
        this.filteredData = this.filterData(this.form.get('algorithms')?.value, "algorithm", this.filteredData);
        if(chartTipe == 'precision'){
          this.filteredData = this.filterRange(this.filteredData, "WKNN", "Mean_error");
        }else{
          this.filteredData = this.filterRange(this.filteredData, "WKNN", "Predicted_points");
        }
        break;
      
      case "DeepLearning":
      case 'DeepLearningPyTorch':
      case 'DeepLearningHybrid':

        break;
      case "SVR":
        this.filteredData = this.filterData(this.form.get('kernels')?.value, "kernel", this.filteredData);
        if(chartTipe == 'precision'){
          this.filteredData = this.filterRange(this.filteredData, "SVR", "Mean_error");
        }else{
          this.filteredData = this.filterRange(this.filteredData, "SVR", "Predicted_points");
        }
        break;
      
      case "NuSVR":
        this.filteredData = this.filterData(this.form.get('kernels')?.value, "kernel", this.filteredData);
        if(chartTipe == 'precision'){
          this.filteredData = this.filterRange(this.filteredData, "NuSVR", "Mean_error");
        }else{
          this.filteredData = this.filterRange(this.filteredData, "NuSVR", "Predicted_points");
        }
        break;
      
      case "LinearSVR":
        if(chartTipe == 'precision'){
          this.filteredData = this.filterRange(this.filteredData, "LinearSVR", "Mean_error");
        }else{
          this.filteredData = this.filterRange(this.filteredData, "LinearSVR", "Predicted_points");
        }
        break;
    }

    if(chartTipe === 'precision'){
      let parsedData = this.parsePrecisionData();
      this.fillPrecisionData(parsedData);
    } 
    else {
      let parsedData = this.parseAccuracyData();
      this.fillAccuracyData(parsedData);
    }
  }

  filterData(filters: any, property: string, data: any[]): any[]{
    const filterArray = Array.isArray(filters) ? filters : [filters];
    return data.filter((object:any) =>{
      const methodDescription = JSON.parse(object.Method_description);
      return filterArray.some((filter: string) => filter === methodDescription[property].toString());
    })
  }

  filterRange(data: any[], rangeType: string, atribute: string): any[] {
    
    return data.map((object: any) => {
      const kRangeStart = this.form.get('kRangeStart')?.value;
      const kRangeEnd = this.form.get('kRangeEnd')?.value;
      const cRangeStart = this.form.get('cRangeStart')?.value;
      const cRangeEnd = this.form.get('cRangeEnd')?.value;
      const gRangeStart = this.form.get('gRangeStart')?.value;
      const gRangeEnd = this.form.get('gRangeEnd')?.value;
      const gamma = this.form.get('gamma')?.value;
      const nuRangeStart = this.form.get('nuRangeStart')?.value;
      const nuRangeEnd = this.form.get('nuRangeEnd')?.value;
      const iRangeStart = this.form.get('iRangeStart')?.value;
      const iRangeEnd = this.form.get('iRangeEnd')?.value;

      const clonedObject = { ...object }
      const objParam = JSON.parse(clonedObject[atribute]);
      const filteredParam: any = {};

      Object.entries(objParam).forEach(([key, value]) => {
        switch(rangeType){
          case 'WKNN':

            const matchesWKNN = parseInt(key.match(/k=(\d+)/)?.[1] || "", 10);

            if (matchesWKNN >= kRangeStart && matchesWKNN <= kRangeEnd) {
              filteredParam[key] = value;
            }
            break;

          case 'SVR':
            const matchesSVR = key.match(/k=C([\d.]+)_G(\w+\.?\w*)/);

            if (matchesSVR) {

                const cValue = parseFloat(matchesSVR[1]);
                let gValue:any = matchesSVR[2];  
        
                const gValueIsNumber = !isNaN(parseFloat(gValue));
        
                if (gValueIsNumber) {
                    gValue = parseFloat(gValue);
                }
        
                let gValueInRange;
                if(gamma === 'range' && gValueIsNumber){
                    gValueInRange = gValue >= gRangeStart && gValue <= gRangeEnd;
                } else if(gamma === 'auto' && gValue === 'auto'){
                    gValueInRange = true;
                }
                else if(gamma === 'scale' && gValue === 'scale'){
                    gValueInRange = true;
                }
        
                if (
                    cValue >= cRangeStart && cValue <= cRangeEnd &&
                    gValueInRange) {
                      filteredParam[key] = value;
                }
            }
            break;
  
          case 'NuSVR':
            const matchesNuSVR = key.match(/k=nu([\d.]+)_C([\d.]+)_G(\w+\.?\w*)/);
            if (matchesNuSVR) {
              const nuValue = parseFloat(matchesNuSVR[1]);
              const cValue = parseFloat(matchesNuSVR[2]);
              let gValue:any = matchesNuSVR[3];  

              const gValueIsNumber = !isNaN(parseFloat(gValue));

              if (gValueIsNumber) {
                gValue = parseFloat(gValue);
              }
        
              let gValueInRange;
              if(gamma === 'range' && gValueIsNumber){
                gValueInRange = gValue >= gRangeStart && gValue <= gRangeEnd;
              } else if(gamma === 'auto' && gValue === 'auto'){
                gValueInRange = true;
              }
              else if(gamma === 'scale' && gValue === 'scale'){
                gValueInRange = true;
              }
        
              if (
                nuValue >= nuRangeStart && nuValue <= nuRangeEnd &&
                cValue >= cRangeStart && cValue <= cRangeEnd &&
                gValueInRange) {
                  filteredParam[key] = value;
              }
            }
            break;

            case 'LinearSVR':
            const matchesLinearSVR = key.match(/k=i([\d.]+)_C(\w+\.?\w*)/);
            if (matchesLinearSVR) {

                const iValue = parseFloat(matchesLinearSVR[1]);
                const cValue = parseFloat(matchesLinearSVR[2]);
        
                if (
                    cValue >= cRangeStart && cValue <= cRangeEnd &&
                    iValue >= iRangeStart && iValue <= iRangeEnd) {
                      filteredParam[key] = value;
                }
            }
            break;
          }
      });

        clonedObject[atribute] = JSON.stringify(filteredParam);
        return clonedObject;
    });
  }
  
  parsePrecisionData() {
    let parsedData = this.filteredData.map((item:any) => {
        let meanErrors = JSON.parse(item.Mean_error);
      
        let data = Object.keys(meanErrors).map(key => ({
            key: key,
            meanError: meanErrors[key]
        }
        ));
      
        const cdfValues = this.calculateCDF(data);

        return {
            ...item,
            PrecisionData: data,
            CdfValues: cdfValues
        };
    });

    return parsedData;
  }

  parseAccuracyData() {
    let parsedData = this.filteredData.map((item: any) => {
      let predictedPoints = JSON.parse(item.Predicted_points);
  
      let predPoints = Object.entries(predictedPoints).map(([key, points]: [string, any]) => ({
        key: key,
        values: points.map(([x, y]: [number, number]) => ({ x, y })),
      }));

      let aleatoryPoints = this.alePoints ? Object.values(this.alePoints).map((point: any) => ({
        x: parseFloat(point.X),
        y: parseFloat(point.Y),
      })) : [];
  
  
      return {
        ...item,
        PredictedData: predPoints,
        AleatoryData: aleatoryPoints,
      };
    });
  
    return parsedData;
  }

  calculateCDF(data: any) {
    const cdfData = [];

    for (let i = 0; i < data.length; i++) {
        let meanErrors = data[i].meanError;
        let cdf:any = [];
        let totalCount = meanErrors.length;
        let cumulativeCount = 0;

        meanErrors.forEach((errorValue:any, index:any) => {
            cumulativeCount += 1;
            const probability = cumulativeCount / totalCount;
            cdf.push({ x: errorValue, y: probability });
        });

        cdf.unshift({ x: 0, y: 0 });

        cdfData.push({
            key: data[i].key,
            cdf: cdf
        });
    }

    return cdfData;
}

  fillPrecisionData(filteredData:any){
    let labels = [];
    this.precisionSummary = [];

    for(let i = 0; i < filteredData.length; i++){
      let item = filteredData[i];
      let cdfValues = item.CdfValues;

      for(let j = 0; j < cdfValues.length; j++){
          let data = cdfValues[j];
          let meanErrors = data.cdf;
          
          let color = this.getSeriesColor(data.key);
          let summary = this.summarizeErrors(
            meanErrors.slice(1).map((point:any) => Number(point.x))
          );
          this.precisionSummary.push({
            label: data.key,
            ...summary
          });

          let dataset = {
              data: meanErrors,
              label: data.key,
              backgroundColor: color.background,
              borderColor: color.border,
              pointBackgroundColor: color.border,
              pointBorderColor: '#fff',
              pointHoverBackgroundColor: '#fff',
              pointHoverBorderColor: color.border,
              fill: false,
              hidden: this.totalSeries.length >= 6,
              borderWidth: 3,
              pointRadius: 2
          }

          for (let selected of this.selectedSeries){
            if (!this.totalSeries.includes(selected) && this.totalSeries.length < 30) {
                this.totalSeries.push(selected);
            }
          }

          if(this.totalSeries.length < 30) {
            this.totalSeries.push(dataset);
          }
          else {
              this.toastr.warning('No se han cargado todas las series. Ajusta tus selecciones o filtros para ver más datos.', 'Límite de 30 series alcanzado', { timeOut: 5000 });
              break;
          }

          const filteredSeries = this.totalSeries.reduce((acc:any, curr:any) => {
            const existingItem = acc.find((item:any) => item.label === curr.label);
          
            if (!existingItem) {
              acc.push(curr);
            }
          
            return acc;
          }, []);

          this.totalSeries = filteredSeries
          labels.push(...meanErrors);
      }
  }

  if(this.totalSeries.length == 0){
    for (let selected of this.selectedSeries){
      if (!this.totalSeries.includes(selected)) {
        this.totalSeries.push(selected)
      }
    }
  }

    this.lineChartData = {
      datasets: this.totalSeries,
      labels: labels
    };

  }

  fillAccuracyData(filteredData:any){
    let colors = ["#00BFFF", "#696969", "#228B22", "#FF69B4", "#DAA520", "#D2691E", "#4682B4"];

    let datasets:any[] = [];
    let labels:any[] = [];
    this.accuracySummary = [];


    let aleatoryPoints = filteredData[0].AleatoryData;
    let pointColors_ale = aleatoryPoints.map(() => '#111827');

    let dataset_ale = {
      data: aleatoryPoints,
      label: "Puntos reales",
      backgroundColor: pointColors_ale,
      borderColor: pointColors_ale,
      pointBackgroundColor: pointColors_ale,
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: pointColors_ale,
      fill: false,
      hidden: false,
      showLine: false,
      pointStyle: 'triangle',
      pointRadius: 8
    }

    datasets.push(dataset_ale);

    for(let i = 0; i < filteredData.length; i++){
      let item = filteredData[i];
      let predictedValues = item.PredictedData;

      for(let j = 0; j < predictedValues.length; j++){
          let data = predictedValues[j];
          let predPoints = data.values;

          let color = this.getSeriesColor(data.key);
          let pointColors = predPoints.map(() => color.border);
          let planarErrors = predPoints
            .slice(0, aleatoryPoints.length)
            .map((point:any, index:number) => Math.hypot(
              point.x - aleatoryPoints[index].x,
              point.y - aleatoryPoints[index].y
            ));
          this.accuracySummary.push({
            label: data.key,
            ...this.summarizeErrors(planarErrors)
          });
          let dataset_pred = {
              data: predPoints,
              label: data.key,
              backgroundColor: pointColors,
              borderColor: pointColors,
              pointBackgroundColor: pointColors,
              pointBorderColor: '#fff',
              pointHoverBackgroundColor: '#fff',
              pointHoverBorderColor: pointColors,
              fill: false,
              hidden: datasets.length > 6,
              showLine: false,
              pointRadius: 6
          }

          if(datasets.length < 30) {
            datasets.push(dataset_pred);
            labels.push(...predPoints);
          }
          else {
              this.toastr.warning('No se han cargado todas las series. Ajusta tus selecciones o filtros para ver más datos.', 'Límite de 30 series alcanzado', { timeOut: 5000 });
              break;
          }
      }
  }

    this.pointsChartData = {
      datasets: datasets,
      labels: labels
    };

  }
  
  public lineChartData: ChartConfiguration['data'] = {
    datasets: [],
    labels: []
  };

  public lineChartOptions: ChartConfiguration['options'] = {
    maintainAspectRatio:false,
    scales: {
      x: {
          title: {
              display: true,
              text: 'Error (m)'
          },
          beginAtZero: true,
          type:"linear",
          grace: '5%',
      },
      y: {
          title: {
              display: true,
              text: 'Distribución acumulada (CDF)'
          },
          type:"linear",
          min: 0,
          max: 1
      },
    },
    plugins:{
      legend:{
        display:true,
        position:"bottom",
        labels:{
          boxWidth:10,
          boxHeight:10,
        },
        onClick: (event, legendItem, legend) => {

            const index = legendItem.datasetIndex;

            const chart = legend.chart;
            const dataset = chart.data.datasets[index||0];

            if (!this.selectedSeries.includes(dataset)) {
              dataset.hidden = false;
              this.selectedSeries.push(dataset);
            }
            else {
              let foundIndex = this.selectedSeries.findIndex((data:any) => data === dataset);
              if (foundIndex !== -1) {
                  this.selectedSeries.splice(foundIndex, 1);
              }
              dataset.hidden = true;
            }

            chart.update(this.selectedSeries);
        }
        
      }
    },
  };

  public lineChartType: ChartType = 'line';


  public pointsChartData: ChartConfiguration['data'] = {
    datasets: [],
    labels: []
  };

  public pointsChartOptions: ChartConfiguration['options'] = {
    maintainAspectRatio:false,
    scales: {
      x: {
          title: {
              display: true,
              text: 'x (m)'
          },
          beginAtZero: true,
          type:"linear",
          grace: '5%',
      },
      y: {
          title: {
              display: true,
              text: 'y (m)'
          },
          type:"linear",
          beginAtZero: true,
          grace: '5%',
      },
    },
    plugins:{
      legend:{
        display:true,
        position:"bottom",
        labels:{
          boxWidth:10,
          boxHeight:10,
        }
      },
    }
  };

  public pointsChartType: ChartType = 'line';

  getSeriesColor(label:string) {
    let hash = 0;
    for (let index = 0; index < label.length; index++) {
      hash = label.charCodeAt(index) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash) % 360;
    return {
      border: 'hsl(' + hue + ', 72%, 42%)',
      background: 'hsla(' + hue + ', 72%, 42%, 0.18)'
    };
  }

  summarizeErrors(errors:number[]) {
    const sorted = errors
      .filter(value => Number.isFinite(value))
      .sort((first, second) => first - second);
    if(sorted.length === 0) {
      return { mean: 0, median: 0, p90: 0, maximum: 0 };
    }

    const percentile = (ratio:number) => {
      const index = Math.min(
        sorted.length - 1,
        Math.max(0, Math.ceil(sorted.length * ratio) - 1)
      );
      return sorted[index];
    };

    return {
      mean: sorted.reduce((total, value) => total + value, 0) / sorted.length,
      median: percentile(0.5),
      p90: percentile(0.9),
      maximum: sorted[sorted.length - 1]
    };
  }

  insertPrecisionData(){
    this.totalSeries = [];
    this.onFilterSelected('precision');
  }

  insertAccuracyData(){
    this.onFilterSelected('accuracy');
  }
}
