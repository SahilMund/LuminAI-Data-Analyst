import { useMutation } from '@tanstack/react-query';
import { AddDataSource, AddDataSourceResponse, DataSources, GetDataSourcesResponse, GetTablesList, GetTablesListResponse, UploadSpreadSheetResponse } from "../interfaces/dataSourceInterface";
import { AxiosError } from 'axios';
import { ApiResponse } from "../interfaces/globalInterfaces";
import { addDataSource, getDataSources, getDataSourceTables, uploadSpreadsheet, uploadDocument } from "../zustand/apis/dataSourceApi";
import dataSetStore from '../zustand/stores/dataSetStore';
// import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { formatDate } from '../utils/dateTimeUtils';

interface ErrorResponse {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

export const useGetDataSourcesMutation = () => {
  const setDataSet = dataSetStore((state) => state.setDataSet);
  return useMutation<ApiResponse<GetDataSourcesResponse>, AxiosError<ErrorResponse>, void>({
    mutationFn: getDataSources,
    onSuccess: (response) => {
      setDataSet(response.data.data_sources)
    },
    onError: (error) => {
      console.log(error.response?.data);
    },
  });
};

export const useAddDataSourceMutation = (resetState: () => void) => {
  const addDataSet = dataSetStore((state) => state.addDataSet);
  return useMutation<ApiResponse<AddDataSourceResponse>, AxiosError<ErrorResponse>, AddDataSource>({
    mutationFn: addDataSource,
    onSuccess: (response) => {
      const newSource: DataSources = {
        id: response.data.id,
        name: response.data.table_name,
        type: 'url',
        connection_url: response.data.connection_url,
        created_at: formatDate(new Date())
      }
      addDataSet(newSource)
      resetState()
      toast.success('Data source added successfully');
    },
    onError: (error) => {
      console.log(error.response?.data);
    },
  });
};

export const useUploadSpreadsheetMutation = () => {
  const addDataSet = dataSetStore((state) => state.addDataSet);
  return useMutation<ApiResponse<UploadSpreadSheetResponse>, AxiosError<ErrorResponse>, File>({
    mutationFn: uploadSpreadsheet,
    onSuccess: (response, file) => {
      const newSource: DataSources = {
        id: response.data.data_source_id,
        name: file.name,
        type: 'spreadsheet',
        table_name: response.data.table_name,
        created_at: formatDate(new Date())
      }
      addDataSet(newSource)
      toast.success(response.message)
    },
    onError: (error) => {
      console.log(error.response?.data.message);
      toast.error(error.response?.data.message)
    },
  });
}

export const useUploadDocumentMutation = () => {
  const addDataSet = dataSetStore((state) => state.addDataSet);
  return useMutation<ApiResponse<UploadSpreadSheetResponse>, AxiosError<ErrorResponse>, File>({
    mutationFn: uploadDocument,
    onSuccess: (response, file) => {
      const newSource: DataSources = {
        id: response.data.data_source_id,
        name: file.name,
        type: 'document',
        table_name: response.data.table_name,
        created_at: formatDate(new Date())
      }
      addDataSet(newSource)
      toast.success(response.message)
    },
    onError: (error) => {
      console.log(error.response?.data.message);
      toast.error(error.response?.data.message)
    },
  });
}

export const useGetTablesMutation = () => {
  const setTables = dataSetStore((state) => state.setTables);
  return useMutation<ApiResponse<GetTablesListResponse>, AxiosError<ErrorResponse>, GetTablesList>({
    mutationFn: getDataSourceTables,
    onSuccess: (response) => {
      setTables(response.data.tables)
      console.log(response.data.tables);
    },
    onError: (error) => {
      console.log(error.response?.data.message);
      toast.error(error.response?.data.message)
    },
  });
}

