import { DATA_SOURCE_ENDPOINTS } from './endPoints';
import { get, post } from './apiClient';
import {
  GetDataSourcesResponse,
  UploadSpreadSheetResponse,
  AddDataSource,
  AddDataSourceResponse,
  GetTablesList,
  GetTablesListResponse
} from '../../interfaces/dataSourceInterface';
import { ApiResponse } from '../../interfaces/globalInterfaces';

type ApiFunction<TInput, TOutput> = (data: TInput) => Promise<ApiResponse<TOutput>>;

export const uploadSpreadsheet: ApiFunction<File, UploadSpreadSheetResponse> = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('table_name', file.name.split('.')[0]);
  return await post(DATA_SOURCE_ENDPOINTS.UPLOAD_SPREADSHEET, formData);
}


export const uploadDocument: ApiFunction<File, UploadSpreadSheetResponse> = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return await post(DATA_SOURCE_ENDPOINTS.UPLOAD_DOCUMENT, formData);
};

export const addDataSource: ApiFunction<AddDataSource, AddDataSourceResponse> = async (data) => {
  return await post(DATA_SOURCE_ENDPOINTS.ADD_DATA_SOURCE, data);
};

export const getDataSources: ApiFunction<void, GetDataSourcesResponse> = async () => {
  return await get(DATA_SOURCE_ENDPOINTS.GET_DATA_SOURCES);
};

export const getDataSourceTables: ApiFunction<GetTablesList, GetTablesListResponse> = async (data) => {
  return await post(DATA_SOURCE_ENDPOINTS.GET_DATA_SOURCE_TABLES, data);
};
