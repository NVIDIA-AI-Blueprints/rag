import { useState, useCallback, useMemo } from "react";
import { Button } from "@kui/react";
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
  
  // Parse array value - wrapped in useMemo to prevent useCallback deps from changing
  const arrayValue = useMemo(() => {
    if (field.type !== "array") return null;
    
    if (Array.isArray(value)) {
      return value;
    }
    
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
    
    // Convert based on array_type
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

  const getInputType = () => {
    switch (field.type) {
      case "integer":
      case "number":
      case "float":
        return "number";
      case "datetime":
        return "datetime-local";
      case "boolean":
        return "checkbox";
      default:
        return "text";
    }
  };

  const getStepValue = () => {
    if (field.type === "float" || field.type === "number") {
      return "0.01";
    }
    if (field.type === "integer") {
      return "1";
    }
    return undefined;
  };

  const displayLabel = `${field.name}${field.required ? " *" : ""} (${field.type}${field.array_type ? `<${field.array_type}>` : ""})`;

  // Boolean field
  if (field.type === "boolean") {
    return (
      <div className="mb-5">
        <label className="flex items-start gap-3 text-xs mb-2">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onChange(field.name, e.target.checked, field.type)}
            className="mt-0.5 rounded border-neutral-600 bg-neutral-700 text-[var(--nv-green)] focus:ring-2 focus:ring-[var(--nv-green)]/50"
          />
          <div className="mb-5">
            <span className="font-medium text-neutral-300">{displayLabel}</span>
            {field.description && (
              <div className="text-neutral-500 mt-1 leading-relaxed">
                {field.description}
              </div>
            )}
          </div>
        </label>
      </div>
    );
  }

  // Array field
  if (field.type === "array") {
    // Special handling for boolean arrays
    if (field.array_type === "boolean") {
      return (
        <div className="mb-5">
          <label className="block text-xs font-medium text-neutral-300 mb-2">
            {displayLabel}
            {field.description && (
              <div className="text-neutral-500 font-normal mt-1 leading-relaxed">
                {field.description}
              </div>
            )}
          </label>
          
          <div className="space-y-2">
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => {
                  const newArray = [...(arrayValue || []), true];
                  onChange(field.name, JSON.stringify(newArray), field.type);
                }}
                className="px-3 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 text-xs font-medium flex items-center gap-1"
              >
                <span>+ Add True</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  const newArray = [...(arrayValue || []), false];
                  onChange(field.name, JSON.stringify(newArray), field.type);
                }}
                className="px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 text-xs font-medium flex items-center gap-1"
              >
                <span>+ Add False</span>
              </button>
            </div>
            
            {arrayValue && arrayValue.length > 0 && (
              <div className="space-y-1">
                {arrayValue.map((item: boolean, index: number) => (
                  <div key={index} className="flex items-center gap-2 p-2 bg-neutral-800 rounded-md">
                    <button
                      type="button"
                      onClick={() => {
                        const newArray = [...arrayValue];
                        newArray[index] = !newArray[index];
                        onChange(field.name, JSON.stringify(newArray), field.type);
                      }}
                      className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                        item 
                          ? 'bg-green-600 text-white hover:bg-green-700' 
                          : 'bg-red-600 text-white hover:bg-red-700'
                      }`}
                    >
                      {item ? 'True' : 'False'}
                    </button>
                    <Button
                      onClick={() => handleArrayRemove(index)}
                      kind="tertiary"
                      color="neutral"
                      size="small"
                      title="Remove item"
                    >
                      <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="#ffffff" strokeWidth="2" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      );
    }

    // Standard array implementation for non-boolean types
    return (
      <div className="mb-5">
        <label className="block text-xs font-medium text-neutral-300 mt-4">
          {displayLabel}
          {field.description && (
            <div className="text-neutral-500 font-normal mt-1 leading-relaxed">
              {field.description}
            </div>
          )}
        </label>
        
        <div className="space-y-2">
          <div className="flex gap-2">
            <input
              type="text"
              value={arrayInputValue}
              onChange={(e) => setArrayInputValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleArrayAdd()}
              placeholder={`Enter ${field.array_type || "text"} value`}
              className="flex-1 rounded-md px-3 py-2 bg-neutral-700 border border-neutral-600 text-sm text-white focus:outline-none focus:ring-2 focus:ring-[var(--nv-green)]/50 focus:border-[var(--nv-green)] transition-colors"
            />
            <Button
              onClick={handleArrayAdd}
              kind="tertiary"
              color="neutral"
              size="small"
              title="Add item"
            >
              <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="#ffffff" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
            </Button>
          </div>
          
          {arrayValue && arrayValue.length > 0 && (
            <div className="space-y-1">
              {arrayValue.map((item: unknown, index: number) => (
                <div key={index} className="flex items-center gap-2 p-2 bg-neutral-800 rounded-md">
                  <span className="flex-1 text-xs text-white">{String(item)}</span>
                  <Button
                    onClick={() => handleArrayRemove(index)}
                    kind="tertiary"
                    color="neutral"
                    size="small"
                    title="Remove item"
                  >
                    <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="#ffffff" strokeWidth="2" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Regular input fields
  return (
    <div className="mb-5">
      <label className="block text-xs font-medium text-neutral-300 mb-2">
        {displayLabel}
        {field.description && (
          <div className="text-neutral-500 font-normal mt-1 leading-relaxed">
            {field.description}
          </div>
        )}
      </label>
      <input
        type={getInputType()}
        value={typeof value === 'string' ? value : String(value || '')}
        onChange={(e) => {
          let processedValue: unknown = e.target.value;
          
          // Convert to proper types based on field type
          switch (field.type) {
            case "datetime":
              // Process datetime to ensure proper format
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
              
            case "string":
            default:
              // Keep as string for string and datetime types
              break;
          }
          
          onChange(field.name, processedValue, field.type);
        }}
        step={getStepValue()}
        maxLength={field.max_length}
        className="w-full rounded-md px-3 py-2 bg-neutral-700 border border-neutral-600 text-sm text-white focus:outline-none focus:ring-2 focus:ring-[var(--nv-green)]/50 focus:border-[var(--nv-green)] transition-colors"
        placeholder={
          field.type === "datetime" ? "YYYY-MM-DDTHH:MM" :
          field.type === "integer" ? "Enter whole number" :
          field.type === "float" || field.type === "number" ? "Enter decimal number" :
          field.max_length ? `Max ${field.max_length} characters` :
          `Enter ${field.type} value`
        }
      />
      
      {field.max_length && field.type === "string" && (
        <div className="text-xs text-neutral-500 mt-1">
          {typeof value === 'string' ? value.length : String(value || '').length}/{field.max_length} characters
        </div>
      )}
    </div>
  );
}; 