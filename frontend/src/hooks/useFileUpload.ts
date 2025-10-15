import { useCallback } from "react";
import { useNewCollectionStore } from "../store/useNewCollectionStore";

/**
 * Custom hook for handling file uploads in the new collection flow.
 * 
 * Provides utilities to handle file selection from drag-and-drop or input elements
 * and automatically adds them to the new collection store.
 * 
 * @returns Object containing file handling functions
 * 
 * @example
 * ```tsx
 * const { handleFileSelect, handleInputChange } = useFileUpload();
 * // Use with input: <input onChange={handleInputChange} />
 * // Use with drag-drop: handleFileSelect(droppedFiles)
 * ```
 */
export const useFileUpload = () => {
  const { addFiles } = useNewCollectionStore();

  const handleFileSelect = useCallback((files: FileList | File[]) => {
    const fileArray = Array.isArray(files) ? files : Array.from(files);
    addFiles(fileArray);
  }, [addFiles]);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFileSelect(e.target.files);
    }
  }, [handleFileSelect]);

  return {
    handleFileSelect,
    handleInputChange,
  };
}; 