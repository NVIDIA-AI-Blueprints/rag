import { useCallback, useState } from "react";
import { useNewCollectionStore } from "../../store/useNewCollectionStore";
import { MetadataField } from "./MetadataField";
import { Button, Card, Flex, Text, Stack } from "@kui/react";
import { ConfirmationModal } from "../modals/ConfirmationModal";
import type { UIMetadataField } from "../../types/collections";

interface FileCardProps {
  file: File;
  index: number;
}

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
    <Card>
      <Flex justify="between" align="center">
        <Text kind="body/regular/sm" style={{ flex: 1, marginRight: 'var(--spacing-density-sm)' }}>
          {file.name}
        </Text>
        <Button
          onClick={handleRemoveClick}
          kind="tertiary"
          color="neutral"
          size="small"
          title="Remove file"
        >
          REMOVE
        </Button>
      </Flex>
      
      {metadataSchema.length > 0 && (
        <Stack gap="density-md" style={{ marginTop: 'var(--spacing-density-sm)' }}>
          {metadataSchema
            .filter((field: UIMetadataField) => field.name !== 'filename')
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
        </Stack>
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
    </Card>
  );
};
