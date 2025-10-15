import { useEffect, useRef } from 'react';
import { FileUploadZone } from './FileUploadZone';
import { FileList } from './FileList';
import { useUploadFileState } from '../../hooks/useUploadFileState';
import { Stack } from '@kui/react';

// Export all upload components for external use
export { FileUploadZone } from './FileUploadZone';
export { FileList } from './FileList';
export { FileItem } from './FileItem';
export { FileMetadataForm } from './FileMetadataForm';

interface NvidiaUploadProps {
  onFilesChange?: (files: File[]) => void;
  onUpload?: (files: File[]) => Promise<void>;
  acceptedTypes?: string[];
  maxFileSize?: number; // in MB
  maxFiles?: number;
  onValidationChange?: (hasInvalidFiles: boolean) => void; // New prop to notify parent about validation state
}

export default function NvidiaUpload({
  onFilesChange,
  acceptedTypes = ['.bmp', '.docx', '.html', '.jpeg', '.json', '.md', '.pdf', '.png', '.pptx', '.sh', '.tiff', '.txt', '.mp3', '.wav'],
  maxFileSize = 400,
  maxFiles = 100,
  onValidationChange
}: NvidiaUploadProps) {
  const { uploadFiles, addFiles, removeFile } = useUploadFileState({
    acceptedTypes,
    maxFileSize,
    maxFiles,
    onFilesChange,
  });

  const previousValidationStateRef = useRef<boolean | null>(null);

  useEffect(() => {
    // Notify parent about validation state changes only when it actually changes
    const hasInvalidFiles = uploadFiles.some(file => file.status === 'error');
    if (previousValidationStateRef.current !== hasInvalidFiles) {
      previousValidationStateRef.current = hasInvalidFiles;
      onValidationChange?.(hasInvalidFiles);
    }
  }, [uploadFiles, onValidationChange]);

  return (
    <Stack gap="density-xl">
      <FileUploadZone
        acceptedTypes={acceptedTypes}
        maxFileSize={maxFileSize}
        onFilesSelected={addFiles}
      />
      
      <FileList
        uploadFiles={uploadFiles}
        onRemoveFile={removeFile}
      />
    </Stack>
  );
} 