import { useState, useEffect } from 'react';
import { BiSend } from 'react-icons/bi';
import { BsStars } from 'react-icons/bs';
import Steps from '../components/steps/Steps';
import SelectDataset from '../components/SelectDataset';
import Avatar from 'react-avatar';
import { useParams } from 'react-router-dom';
import { useStreamChat } from '../hooks/useChat';
import { ConversationMessages, ProcessingMessage } from '../interfaces/chatInterface';
import TypewriterWithHighlight from '../components/TypewriterWithHighlight';
import ChartComponent from '../components/ChartComponent';
// import { toast } from 'react-toastify';
import dataSetStore from '../zustand/stores/dataSetStore';

export default function Chat() {

  // const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [question, setQuestion] = useState('');
  const [datasetType, setDatasetType] = useState<string>();
  const [messages, setMessages] = useState<ConversationMessages[]>([]);
  const [processingMessages, setProcessingMessages] = useState<ProcessingMessage[]>([]);

  const tables = dataSetStore((state) => state.tables);
  const dataSets = dataSetStore((state) => state.dataSets);
  const setTables = dataSetStore((state) => state.setTables);
  const selectedModel = dataSetStore((state) => state.selectedModel);
  // Get the data_source_id from URL parameters
  const { data_source_id, conversation_id } = useParams();
  const {
    mutate: sendMessage,
    // streamData,
    status,
    // error,
  } = useStreamChat({
    onStreamData: (data) => {
      setProcessingMessages(data);
      console.log('Received data:', data);
    },
    onSuccess: (data) => {
      const ai_answer: any = {}
      data.forEach((message: any) => {
        if (message.answer) {
          ai_answer["answer"] = message.answer
        }
        if (message.formatted_data_for_visualization) {
          ai_answer["formatted_data_for_visualization"] = message.formatted_data_for_visualization
        }
        if (message.recommended_visualization) {
          ai_answer["recommended_visualization"] = message.recommended_visualization
        }
        if (message.source_documents) {
          ai_answer["source_documents"] = message.source_documents
        }
      })
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          ai_answer: ai_answer
        },
      ]);
    }
  });

  useEffect(() => {
    const filteredDataSet = dataSets?.filter((dataSet) => dataSet.id === Number(data_source_id)) || [];
    if (filteredDataSet.length > 0) {
      setTables([filteredDataSet[0]?.table_name || ''])
      setDatasetType(filteredDataSet[0]?.type)
    }
  }, [data_source_id])


  const askQuestion = () => {
    // Safely spread messages, handling empty array case
    setMessages((prevMessages) => [...prevMessages, { user_question: question }]);
    setProcessingMessages([]);
    // Clear the question input after sending
    setQuestion('');
    sendMessage({
      question: question,
      type: datasetType || 'url',
      conversaction_id: Number(conversation_id),
      dataset_id: Number(data_source_id),
      selected_tables: tables,
      llm_model: selectedModel
    })
  };

  console.log(selectedModel);
  console.log("TABLES", tables);

  return (
    <div className="flex flex-col py-8 pr-8 h-screen">
      <div className="h-full rounded-[20px] bg-white border border-blue-gray-100 dark:bg-maroon-400 dark:border-maroon-600 flex justify-between w-full">
        {/* answer div */}
        {/* <div className={`mt-auto flex flex-col p-8 ${messages?.length > 0 ? 'w-1/2' : 'w-full'}`}> */}
        <div className={`flex flex-col h-full ${messages?.length > 0 ? 'w-1/2' : 'w-full'}`}>
          {/* Scrollable messages container */}
          <div className="flex-1 overflow-y-auto p-8 space-y-6">
            {messages?.length > 0 ? (
              messages.map((message, index) => (
                <div key={index}>
                  {/* User question */}
                  {message?.user_question && (
                    <div className="mb-6 flex items-start">
                      <Avatar
                        name="Spandan Joshi"
                        size="40"
                        className="h-8 w-8 mr-2 rounded-md flex-shrink-0"
                      />
                      <p className="text-md text-navy-600">{message?.user_question}</p>
                    </div>
                  )}

                  {/* AI bot reply */}
                  {message.ai_answer && (
                    <div className="mb-6 bg-blue-gray-50 rounded-lg p-4">
                      <div className="flex items-start mb-4">
                        <BsStars className="text-3xl text-navy-600 mr-2 flex-shrink-0" />
                        <TypewriterWithHighlight text={message.ai_answer.answer || ""} />
                      </div>

                      {message.ai_answer.source_documents && message.ai_answer.source_documents.length > 0 && (
                        <div className="mt-4 border-t border-gray-200 pt-3">
                          <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Sources:</p>
                          <div className="flex flex-wrap gap-2">
                            {message.ai_answer.source_documents.map((doc: any, docIndex: number) => (
                              <div key={docIndex} className="text-[10px] bg-white text-navy-600 px-2 py-1 rounded border border-blue-100 flex items-center shadow-sm">
                                <span className="font-medium truncate max-w-[150px]">{doc.metadata?.source || `Doc ${docIndex + 1}`}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {
                        message?.ai_answer?.formatted_data_for_visualization && (
                          <div className="w-full h-[400px]">
                            <ChartComponent
                              type={message?.ai_answer?.recommended_visualization || ""}
                              data={message?.ai_answer?.formatted_data_for_visualization}
                            />
                          </div>
                        )
                      }

                    </div>
                  )}
                </div>
              ))
            ) : (
              <SelectDataset />
            )}
          </div>
          {/* textarea div */}
          <div className="mt-auto flex justify-center items-center">
            <div className={`relative ${messages?.length > 0 ? 'w-full mx-4 bg-white' : 'w-3/5'}`}>
              <textarea
                className={`w-full ${messages?.length > 0 ? 'h-20' : 'h-40'} p-4 bg-blue-gray-50 border-blue-gray-100 text-navy-600 placeholder-gray-400 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent`}
                placeholder="Type Your Question..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />
              <button
                onClick={() => askQuestion()}
                className={`absolute bottom-4 right-4 border border-blue-gray-100 bg-white text-navy-800 p-2 rounded-lg transition-colors`}
              >
                <BiSend className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
        {/* Processing div step */}
        {messages?.length > 0 && (
          <Steps processingMessages={processingMessages} isProcessing={status === "pending"} />
        )}
      </div>
    </div>
  );
}
