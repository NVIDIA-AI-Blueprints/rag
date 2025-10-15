import { useCallback, useState } from "react";
import { useNewCollectionStore } from "../../store/useNewCollectionStore";
import { MetadataField } from "./MetadataField";
import { Button } from "@kui/react";
import { ConfirmationModal } from "../modals/ConfirmationModal";
import type { UIMetadataField } from "../../types/collections";

interface FileCardProps {
  file: File;
  index: number;
}

// No icon needed, using text instead

export const FileCard = ({ file, index }: FileCardProps) => {
  const { metadataSchema, fileMetadata, removeFile, updateMetadataField } = useNewCollectionStore();
  const [showRemoveModal, setShowRemoveModal] = useState(false);

  const handleRemoveClick = useCallback(() => {
    setShowRemoveModal(true);
  }, []);

  const handleConfirmRemove = useCallback(() => {
    removeFile(index);
    setShowRemoveModal(false);
  }, [index, removeFile]);

  const handleMetadataChange = useCallback((fieldName: string, value: unknown) => {
    updateMetadataField(file.name, fieldName, value);
  }, [file.name, updateMetadataField]);

  return (
    <div className="p-3 bg-neutral-800 rounded-md">
      <div className="flex justify-between items-center">
        <span className="text-sm text-white truncate flex-1 mr-2">{file.name}</span>
        <Button
          onClick={handleRemoveClick}
          kind="tertiary"
          color="neutral"
          size="small"
          title="Remove file"
        >
          REMOVE
        </Button>
      </div>
      
      {metadataSchema.length > 0 && (
        <div className="mt-2 space-y-4">
          {metadataSchema
            .filter((field: UIMetadataField) => field.name !== 'filename') // Filter out filename field
            .map((field: UIMetadataField) => (
            <MetadataField
              key={field.name}
              fileName={file.name}
              field={field}
              value={(() => {
                const existingValue = fileMetadata[file.name]?.[field.name];
                if (existingValue !== undefined) return existingValue;
                switch (field.type) {
                  case "boolean": return false;
                  case "array": return [];
                  case "integer":
                  case "float":
                  case "number": return null;
                  default: return "";
                }
              })()}
              onChange={handleMetadataChange}
            />
          ))}
        </div>
      )}
      
      <ConfirmationModal
        isOpen={showRemoveModal}
        onClose={() => setShowRemoveModal(false)}
        onConfirm={handleConfirmRemove}
        title="Remove File"
        message={`Are you sure you want to remove "${file.name}" from this collection?`}
        confirmText="Remove"
        confirmColor="danger"
      />
    </div>
  );
}; 