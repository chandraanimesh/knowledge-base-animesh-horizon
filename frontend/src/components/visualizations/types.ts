export interface AxisSpec {
  label: string;
  unit?: string | null;
}

export interface BarDataPoint {
  label: string;
  value: number;
}

export interface BarSeries {
  name: string;
  data: BarDataPoint[];
}

export interface BarVisualization {
  type: 'bar';
  title: string;
  x_axis: AxisSpec;
  y_axis: AxisSpec;
  series: BarSeries[];
}

export interface LineDataPoint {
  label: string;
  value: number;
}

export interface LineSeries {
  name: string;
  data: LineDataPoint[];
}

export interface LineVisualization {
  type: 'line';
  title: string;
  x_axis: AxisSpec;
  y_axis: AxisSpec;
  series: LineSeries[];
}

export interface PieDataPoint {
  label: string;
  value: number;
}

export interface PieVisualization {
  type: 'pie';
  title: string;
  unit?: string | null;
  data: PieDataPoint[];
}

export interface AreaDataPoint {
  label: string;
  value: number;
}

export interface AreaSeries {
  name: string;
  data: AreaDataPoint[];
}

export interface AreaVisualization {
  type: 'area';
  title: string;
  x_axis: AxisSpec;
  y_axis: AxisSpec;
  series: AreaSeries[];
}

export type TableColumnType = 'string' | 'number' | 'integer' | 'percent' | 'date';

export interface TableColumn {
  key: string;
  label: string;
  type: TableColumnType;
  unit?: string | null;
}

export interface TableVisualization {
  type: 'table';
  title: string;
  columns: TableColumn[];
  rows: Record<string, any>[];
}

export type NetworkNodeType =
  | 'company'
  | 'person'
  | 'department'
  | 'document'
  | 'topic'
  | 'deal'
  | 'event'
  | 'other';

export interface NetworkNode {
  id: string;
  label: string;
  type: NetworkNodeType;
}

export interface NetworkEdge {
  source: string;
  target: string;
  label: string;
}

export interface NetworkVisualization {
  type: 'network';
  title: string;
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

export interface TimelineEvent {
  date: string;
  label: string;
  description: string;
}

export interface TimelineVisualization {
  type: 'timeline';
  title: string;
  events: TimelineEvent[];
}

export type VisualizationSpec =
  | BarVisualization
  | LineVisualization
  | PieVisualization
  | AreaVisualization
  | TableVisualization
  | NetworkVisualization
  | TimelineVisualization;
