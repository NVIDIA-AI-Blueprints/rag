import { useState, useCallback, useMemo } from "react";
import { Button, FormField, TextInput, Switch, Flex, Text, Tag } from "@kui/react";
import { X, Plus } from "lucide-react";
import type { UIMetadataField } from "../../types/collections";

interface MetadataFieldProps {
  fileName: string;
  field: UIMetadataField;
  value: unknown;
  onChange: (fieldName: string, value: unknown, fieldType: string) => void;
}

export const MetadataField = ({ 
  field, 
  value, 
  onChange 
}: MetadataFieldProps) => {
  const [arrayInputValue, setArrayInputValue] = useState("");
  
  const arrayValue = useMemo(() => {
    if (field.type !== "array") return null;
    if (Array.isArray(value)) return value;
    if (typeof value === "string") {
      try {
        return JSON.parse(value || "[]");
      } catch {
        return [];
      }
    }
    return [];
  }, [field.type, value]);

  const handleArrayAdd = useCallback(() => {
    if (!arrayInputValue.trim() || !arrayValue) return;
    
    let processedValue: string | number | boolean = arrayInputValue.trim();
    
    if (field.array_type === "integer") {
      const num = parseInt(processedValue);
      if (isNaN(num)) return;
      processedValue = num;
    } else if (field.array_type === "float" || field.array_type === "number") {
      const num = parseFloat(processedValue);
      if (isNaN(num)) return;
      processedValue = num;
    } else if (field.array_type === "boolean") {
      processedValue = processedValue.toLowerCase() === "true" || processedValue === "1";
    }
    
    const newArray = [...arrayValue, processedValue];
    onChange(field.name, newArray, field.type);
    setArrayInputValue("");
  }, [arrayInputValue, arrayValue, field, onChange]);

  const handleArrayRemove = useCallback((index: number) => {
    if (!arrayValue) return;
    const newArray = arrayValue.filter((_: unknown, i: number) => i !== index);
    onChange(field.name, newArray, field.type);
  }, [arrayValue, field, onChange]);

  const getInputType = (): "number" | "text" | "search" | "tel" | "url" | "email" | "password" | undefined => {
    switch (field.type) {
      case "integer":
      case "number":
      case "float":
        return "number";
      case "datetime":
        return "text"; // datetime fields handled separately
      default:
        return "text";
    }
  };

  const displayLabel = `${field.name}${field.required ? " *" : ""} (${field.type}${field.array_type ? `<${field.array_type}>` : ""})`;

  // Boolean field
  if (field.type === "boolean") {
    return (
      <FormField slotLabel={displayLabel} slotHelp={field.description}>
        <Flex align="center" gap="density-sm">
          <Switch
            checked={Boolean(value)}
            onCheckedChange={(checked) => onChange(field.name, checked, field.type)}
          />
          <Text kind="body/regular/sm">{value ? 'Yes' : 'No'}</Text>
        </Flex>
      </FormField>
    );
  }

  // Array field
  if (field.type === "array") {
    if (field.array_type === "boolean") {
      return (
        <FormField slotLabel={displayLabel} slotHelp={field.description}>
          <Flex gap="density-sm" style={{ marginBottom: 'var(--spacing-density-sm)' }}>
            <Button
              onClick={() => {
                const newArray = [...(arrayValue || []), true];
                onChange(field.name, JSON.stringify(newArray), field.type);
              }}
              kind="tertiary"
              color="brand"
              size="small"
            >
              + Add True
            </Button>
            <Button
              onClick={() => {
                const newArray = [...(arrayValue || []), false];
                onChange(field.name, JSON.stringify(newArray), field.type);
              }}
              kind="tertiary"
              color="neutral"
              size="small"
            >
              + Add False
            </Button>
          </Flex>
          
          {arrayValue && arrayValue.length > 0 && (
            <Flex gap="density-xs" style={{ flexWrap: 'wrap' }}>
              {arrayValue.map((item: boolean, index: number) => (
                <Tag
                  key={index}
                  color={item ? "green" : "red"}
                  kind="solid"
                  density="compact"
                  onClick={() => handleArrayRemove(index)}
                  style={{ cursor: 'pointer' }}
                >
                  {item ? 'True' : 'False'} <X size={12} />
                </Tag>
              ))}
            </Flex>
          )}
        </FormField>
      );
    }

    return (
      <FormField slotLabel={displayLabel} slotHelp={field.description}>
        <Flex gap="density-sm" align="center" style={{ marginBottom: 'var(--spacing-density-sm)' }}>
          <TextInput
            value={arrayInputValue}
            onValueChange={setArrayInputValue}
            onKeyDown={(e) => e.key === "Enter" && handleArrayAdd()}
            placeholder={`Enter ${field.array_type || "text"} value`}
            style={{ flex: 1 }}
          />
          <Button
            onClick={handleArrayAdd}
            kind="tertiary"
            color="brand"
            size="small"
            title="Add item"
          >
            <Plus size={16} />
          </Button>
        </Flex>
        
        {arrayValue && arrayValue.length > 0 && (
          <Flex gap="density-xs" style={{ flexWrap: 'wrap' }}>
            {arrayValue.map((item: unknown, index: number) => (
              <Tag
                key={index}
                color="gray"
                kind="outline"
                density="compact"
                onClick={() => handleArrayRemove(index)}
                style={{ cursor: 'pointer' }}
              >
                {String(item)} <X size={12} />
              </Tag>
            ))}
          </Flex>
        )}
      </FormField>
    );
  }

  // Regular input fields
  return (
    <FormField 
      slotLabel={displayLabel} 
      slotHelp={field.description || (field.max_length ? `Max ${field.max_length} characters` : undefined)}
    >
      <TextInput
        type={getInputType()}
        value={typeof value === 'string' ? value : String(value || '')}
        onValueChange={(newValue) => {
          let processedValue: unknown = newValue;
          
          switch (field.type) {
            case "datetime":
              if (processedValue && typeof processedValue === "string" && processedValue.length === 16) {
                processedValue = `${processedValue}:00`;
              }
              break;
            case "integer":
              if (processedValue && typeof processedValue === "string") {
                const numValue = parseInt(processedValue.trim());
                processedValue = isNaN(numValue) ? processedValue : numValue;
              }
              break;
            case "float":
            case "number":
              if (processedValue && typeof processedValue === "string") {
                const numValue = parseFloat(processedValue.trim());
                processedValue = isNaN(numValue) ? processedValue : numValue;
              }
              break;
          }
          
          onChange(field.name, processedValue, field.type);
        }}
        placeholder={
          field.type === "datetime" ? "YYYY-MM-DDTHH:MM" :
          field.type === "integer" ? "Enter whole number" :
          field.type === "float" || field.type === "number" ? "Enter decimal number" :
          `Enter ${field.type} value`
        }
      />
    </FormField>
  );
};
