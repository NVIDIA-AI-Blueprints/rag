import { useCallback } from "react";
import type { UIMetadataField, MetadataFieldType } from "../../types/collections";
import { 
  Button, 
  Text, 
  Flex, 
  Badge, 
  Divider
} from "@kui/react";
import { useCollectionsStore } from "../../store/useCollectionsStore";

/**
 * Type guard to check if a value is a valid MetadataFieldType
 */
const isValidMetadataFieldType = (type: unknown): type is MetadataFieldType => {
  const validTypes: MetadataFieldType[] = [
    "string", 
    "integer", 
    "float", 
    "number", 
    "boolean", 
    "datetime", 
    "array"
  ];
  return typeof type === "string" && validTypes.includes(type as MetadataFieldType);
};

interface FieldDisplayCardProps {
  field: UIMetadataField;
  onEdit: () => void;
  onDelete: () => void;
}

const EditIcon = () => (
  <svg 
    className="w-4 h-4" 
    fill="none" 
    stroke="currentColor" 
    viewBox="0 0 24 24"
    data-testid="edit-icon"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
  </svg>
);

const DeleteIcon = () => (
  <svg 
    className="w-4 h-4" 
    fill="none" 
    stroke="currentColor" 
    viewBox="0 0 24 24"
    data-testid="delete-icon"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
  </svg>
);

export const FieldDisplayCard = ({ field, onEdit, onDelete }: FieldDisplayCardProps) => {
  const { getFieldTypeColor } = useCollectionsStore();
  
  const handleEdit = useCallback(() => {
    onEdit();
  }, [onEdit]);

  const handleDelete = useCallback(() => {
    onDelete();
  }, [onDelete]);

  // Validate field.type and get appropriate color
  const fieldTypeColor = (() => {
    if (isValidMetadataFieldType(field.type)) {
      return getFieldTypeColor(field.type);
    } else {
      // Log warning for debugging purposes
      console.warn(`Invalid field type encountered: "${field.type}" for field "${field.name}". Using default color.`);
      // Return default color for invalid types
      return "gray";
    }
  })();

  return (
    <>
    <Flex justify="between" align="start" padding="density-lg">
      <Flex 
        align="center" 
        gap="3" 
        style={{ flex: 1 }}
        data-testid="field-content"
      >
        <Text 
          kind="body/bold/md"
          data-testid="field-name"
        >
          {field.name}
        </Text>
          <Badge 
           kind="solid"
           color={fieldTypeColor}
           data-testid="field-type"
         >
           {field.type}
         </Badge>
      </Flex>
      
      <Flex 
        align="center" 
        gap="2"
        data-testid="field-actions"
      >
        <Button 
          kind="tertiary"
          size="tiny"
          onClick={handleEdit} 
          title="Edit"
          data-testid="edit-button"
        >
          <EditIcon />
        </Button>
        <Button 
          kind="tertiary"
          size="tiny"
          onClick={handleDelete} 
          title="Delete"
          data-testid="delete-button"
          style={{ color: 'var(--text-color-danger)' }}
        >
          <DeleteIcon />
        </Button>
      </Flex>
    </Flex>
    <Divider />
    </>
  );
}; 