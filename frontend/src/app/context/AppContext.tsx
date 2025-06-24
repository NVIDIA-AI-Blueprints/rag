// AppContext.tsx

"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useRef,
  ReactNode,
} from "react";
import { Document } from "@/types/documents";
import { Collection } from "@/types/collections";
import { ChatMessage } from "@/types/chat";

interface AppState {
  collections: Collection[];
  selectedCollections: string[]; // CHANGED
  documents: Document[];
  chatMessages: ChatMessage[];
  loading: boolean;
  error: string | null;
}

export interface IngestionTask {
  id: string;
  collection_name: string;
  state: "PENDING" | "FINISHED" | "FAILED" | "UNKNOWN";
  created_at: string;
  documents?: string[];
  result?: any;
}

interface AppContextType extends AppState {
  setCollections: (collections: Collection[]) => void;
  setSelectedCollections: React.Dispatch<React.SetStateAction<string[]>>;
  setDocuments: (documents: Document[]) => void;
  addChatMessage: (message: ChatMessage) => void;
  clearChatMessages: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  pendingTasks: IngestionTask[];
  addPendingTask: (task: IngestionTask) => void;
  updateTaskStatus: (
    taskId: string,
    state: IngestionTask["state"],
    result?: any
  ) => void;
  removePendingTask: (taskId: string) => void;
  onDocumentsUpdated: (collectionName: string, callback: () => void) => () => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]); // CHANGED
  const [documents, setDocuments] = useState<Document[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingTasks, setPendingTasks] = useState<IngestionTask[]>([]);

  const documentUpdateCallbacksRef = useRef<{
    [collectionName: string]: Array<() => void>;
  }>({});

  const addChatMessage = (message: ChatMessage) => {
    setChatMessages((prev) => [...prev, message]);
  };

  const clearChatMessages = () => setChatMessages([]);

  const addPendingTask = (task: IngestionTask) =>
    setPendingTasks((prev) => [...prev, task]);

  const updateTaskStatus = (
    taskId: string,
    state: IngestionTask["state"],
    result?: any
  ) => {
    setPendingTasks((prev) => {
      const updatedTasks = prev.map((task) =>
        task.id === taskId
          ? { ...task, state, result: result || task.result }
          : task
      );

      const updatedTask = updatedTasks.find((task) => task.id === taskId);
      if (
        updatedTask &&
        (state === "FINISHED" || state === "FAILED") &&
        updatedTask.collection_name
      ) {
        const cbs = documentUpdateCallbacksRef.current[updatedTask.collection_name];
        if (cbs) {
          setTimeout(() => {
            cbs.forEach((cb) => {
              try {
                cb();
              } catch (e) {
                console.error("Error executing document update callback:", e);
              }
            });
          }, 0);
        }
      }

      return updatedTasks;
    });
  };

  const removePendingTask = (taskId: string) =>
    setPendingTasks((prev) => prev.filter((task) => task.id !== taskId));

  const onDocumentsUpdated = (collectionName: string, callback: () => void) => {
    if (!documentUpdateCallbacksRef.current[collectionName]) {
      documentUpdateCallbacksRef.current[collectionName] = [];
    }
    documentUpdateCallbacksRef.current[collectionName].push(callback);
    return () => {
      documentUpdateCallbacksRef.current[collectionName] = documentUpdateCallbacksRef
        .current[collectionName]
        .filter((cb) => cb !== callback);
    };
  };

  useEffect(() => {
    if (pendingTasks.length === 0) return;

    const intervalId = setInterval(async () => {
      const active = pendingTasks.filter((t) => t.state === "PENDING");
      if (active.length === 0) return;

      for (const task of active) {
        try {
          const res = await fetch(`/api/task-status?task_id=${task.id}`);
          if (!res.ok) continue;

          const data = await res.json();
          if (task.state !== data.state) {
            updateTaskStatus(task.id, data.state, data.result);

            if (["FINISHED", "FAILED"].includes(data.state)) {
              const resp = await fetch("/api/collections");
              if (resp.ok) {
                const { collections: updated } = await resp.json();
                const formatted = updated.map((c: any) => ({
                  collection_name: c.collection_name,
                  document_count: c.num_entities,
                  index_count: c.num_entities,
                  metadata_schema: c.metadata_schema ?? [],
                }));

                if (data.result.failed_documents?.length > 0) {
                  localStorage.setItem(
                    `failedDocs:${task.collection_name}`,
                    JSON.stringify(data.result.failed_documents)
                  );
                }

                const changed =
                  JSON.stringify(formatted) !== JSON.stringify(collections);
                if (changed) setCollections(formatted);
              }
            }
          }
        } catch (err) {
          console.error(`Polling error for ${task.id}:`, err);
        }
      }
    }, 5000);

    return () => clearInterval(intervalId);
  }, [pendingTasks, collections]);

  return (
    <AppContext.Provider
      value={{
        collections,
        selectedCollections, // CHANGED
        documents,
        chatMessages,
        loading,
        error,
        setCollections,
        setSelectedCollections, // CHANGED
        setDocuments,
        addChatMessage,
        clearChatMessages,
        setLoading,
        setError,
        pendingTasks,
        addPendingTask,
        updateTaskStatus,
        removePendingTask,
        onDocumentsUpdated,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used within an AppProvider");
  return context;
};
