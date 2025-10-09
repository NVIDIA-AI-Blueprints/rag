// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React, { useCallback, useMemo, useRef } from "react";
import type { Filter } from "../../types/chat";
import type { MetadataFieldType, Collection, APIMetadataField } from "../../types/collections";
import { useCollections } from "../../api/useCollectionsApi";
import { useMetadataValues } from "../../api/useMetadataValues";
import { useCollectionsStore } from "../../store/useCollectionsStore";
import { Block, Flex, Text, TextInput, Tag, Popover, Button } from "@kui/react";

// Modern ESNext interfaces with proper typing
interface Props {
  filters: Filter[];
  setFilters: (filters: Filter[]) => void;
}

interface FilterState {
  stage: "field" | "op" | "value";
  input: string;
  draft: { field?: string; op?: string; value?: string };
  selectedValues: string[];
  activeIdx: number;
  show: boolean;
}

interface FieldMeta {
  map: Record<string, MetadataFieldType>;
  arrayTypeMap: Record<string, string>;
}

// Constants for maintainable styling
const STYLES = {
  container: {
    position: 'relative' as const,
    backgroundColor: 'var(--background-color-surface)',
    border: '1px solid var(--border-color-subtle)',
    borderRadius: '8px',
    minHeight: '48px',
  },
  flex: {
    flexWrap: 'wrap' as const,
    minHeight: '48px',
  },
} as const;

// Operators exactly matching backend support (from metadata_validation.py)
const OPERATORS_BY_TYPE: Record<MetadataFieldType, readonly string[]> = {
  string: ["=", "!=", "like", "in", "not in"],
  integer: ["=", "!=", ">", "<", ">=", "<=", "in", "not in"],
  float: ["=", "!=", ">", "<", ">=", "<=", "in", "not in"],
  number: ["=", "!=", ">", "<", ">=", "<=", "in", "not in"],
  boolean: ["=", "!="],
  datetime: ["before", "after", "=", "!=", ">", "<", ">=", "<="],
  array: ["array_contains", "array_contains_all", "array_contains_any", "includes", "does not include", "=", "!=", "in", "not in"],
} as const;

// Utility functions using modern ESNext syntax
const convertValueToType = (
  value: string, 
  fieldType: MetadataFieldType = "string", 
  operator?: string,
  arrayTypeMap?: Record<string, string>,
  fieldName?: string
): string | number | boolean | (string | number | boolean)[] => {
  const trimmedValue = value.trim();
  
  // Special handling for array operators using nullish coalescing
  if (operator && ['array_contains_all', 'array_contains_any', 'in', 'not in'].includes(operator)) {
    try {
      if (trimmedValue.startsWith('[') && trimmedValue.endsWith(']')) {
        return JSON.parse(trimmedValue);
      }
      const items = trimmedValue.split(',').map(item => item.trim()).filter(Boolean);
      return items.length > 0 ? items : [trimmedValue];
    } catch {
      return [trimmedValue];
    }
  }
  
  switch (fieldType) {
    case "integer": {
      const intValue = parseInt(trimmedValue, 10);
      return Number.isNaN(intValue) ? trimmedValue : intValue;
    }
    case "float":
    case "number": {
      const numValue = parseFloat(trimmedValue);
      return Number.isNaN(numValue) ? trimmedValue : numValue;
    }
    case "boolean": {
      const lowerValue = trimmedValue.toLowerCase();
      return lowerValue === "true" ? true : lowerValue === "false" ? false : trimmedValue;
    }
    case "array": {
      if (operator) {
        // Special case: array_contains expects a single scalar value (not an array)
        if (operator === 'array_contains') {
          const arrayElementType = fieldName ? arrayTypeMap?.[fieldName] : undefined;
          switch (arrayElementType) {
            case "integer": {
              const intValue = parseInt(trimmedValue, 10);
              return Number.isNaN(intValue) ? trimmedValue : intValue;
            }
            case "float":
            case "number": {
              const numValue = parseFloat(trimmedValue);
              return Number.isNaN(numValue) ? trimmedValue : numValue;
            }
            case "boolean": {
              const lowerItem = trimmedValue.toLowerCase();
              return lowerItem === "true" ? true : lowerItem === "false" ? false : trimmedValue;
            }
            default:
              return trimmedValue;
          }
        }
        try {
          if (trimmedValue.startsWith('[') && trimmedValue.endsWith(']')) {
            return JSON.parse(trimmedValue);
          }
          const items = trimmedValue.split(',').map(item => item.trim()).filter(Boolean);
          const arrayElementType = fieldName ? arrayTypeMap?.[fieldName] : undefined;
          
          if (arrayElementType) {
            return items.map(item => {
              switch (arrayElementType) {
                case "integer": {
                  const intValue = parseInt(item, 10);
                  return Number.isNaN(intValue) ? item : intValue;
                }
                case "float":
                case "number": {
                  const numValue = parseFloat(item);
                  return Number.isNaN(numValue) ? item : numValue;
                }
                case "boolean": {
                  const lowerItem = item.toLowerCase();
                  return lowerItem === "true" ? true : lowerItem === "false" ? false : item;
                }
                default:
                  return item;
              }
            });
          }
          return items;
        } catch {
          return [trimmedValue];
        }
      }
      return trimmedValue;
    }
    default:
      return trimmedValue;
  }
};

const formatFilterValue = (value: unknown): string => {
  if (Array.isArray(value)) {
    return value.length > 0 
      ? `[${value.map(v => `"${String(v)}"`).join(', ')}]`
      : '[]';
  }
  return typeof value === 'string' ? `"${value}"` : String(value);
};

// Custom hooks for state management
const useFilterState = (_initialFilters: Filter[]) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [filterState, setFilterState] = React.useState<FilterState>({
    stage: "field",
    input: "",
    draft: {},
    selectedValues: [],
    activeIdx: 0,
    show: false,
  });

  const updateState = useCallback((updates: Partial<FilterState>) => {
    setFilterState(prev => ({ ...prev, ...updates }));
  }, []);

  const resetToField = useCallback(() => {
    setFilterState({
      stage: "field",
      input: "",
      draft: {},
      selectedValues: [],
      activeIdx: 0,
      show: false,
    });
  }, []);

  return {
    ...filterState,
    updateState,
    resetToField,
    inputRef,
    setFilterState,
  };
};

const useFieldMeta = (collections: Collection[], selectedCollections: string[]) => {
  return useMemo((): FieldMeta => {
    const map: Record<string, MetadataFieldType> = {};
    const arrayTypeMap: Record<string, string> = {};
    
    const sourceCollections = selectedCollections.length > 0
      ? collections.filter(c => selectedCollections.includes(c.collection_name))
      : collections;
    
    sourceCollections.forEach(col => {
      col.metadata_schema?.forEach((field: APIMetadataField) => {
        map[field.name] = field.type ?? "string";
        if (field.array_type) {
          arrayTypeMap[field.name] = field.array_type;
        }
      });
    });

    return { map, arrayTypeMap };
  }, [collections, selectedCollections]);
};

const useFilterSuggestions = (
  stage: FilterState['stage'],
  input: string,
  fieldMeta: FieldMeta,
  draft: FilterState['draft'],
  valueOptions: string[],
  selectedValues: string[],
  availableFields: string[]
) => {
  return useMemo((): string[] => {
    switch (stage) {
      case "field": {
        const term = input.trim().toLowerCase();
        return term 
          ? availableFields.filter(f => f.toLowerCase().includes(term))
          : availableFields;
      }
      case "op": {
        if (!draft.field) return [];
        const fieldType = fieldMeta.map[draft.field] ?? "string";
        return [...(OPERATORS_BY_TYPE[fieldType] ?? [])];
      }
      case "value": {
        if (!draft.field || !draft.op) return [];
        const fieldType = fieldMeta.map[draft.field] ?? "string";
        if (fieldType === "datetime") return [];
        
        const filtered = valueOptions.filter(v => 
          v.toLowerCase().includes(input.toLowerCase())
        );
        
        const isArrayField = fieldType === "array";
        const shouldUseMultiSelection = isArrayField;
        
        return shouldUseMultiSelection 
          ? filtered.filter(v => !selectedValues.includes(v))
          : filtered;
      }
      default:
        return [];
    }
  }, [stage, input, fieldMeta, draft, valueOptions, selectedValues, availableFields]);
};

export default function FilterBar({ filters, setFilters }: Props) {
  const { data: collections = [], isLoading: isLoadingCollections, error: collectionsError } = useCollections();
  const { selectedCollections } = useCollectionsStore();

  const filterState = useFilterState(filters);
  const { stage, input, draft, selectedValues, activeIdx, show, updateState, resetToField, inputRef } = filterState;

  // Fetch distinct metadata values for current field across selected collections
  const collectionsToQuery = selectedCollections.length > 0 
    ? selectedCollections 
    : collections.map((c: Collection) => c.collection_name);
    
  const { data: valueOptions = [], isLoading: valuesLoading } = useMetadataValues(
    collectionsToQuery,
    draft.field ?? ""
  );

  const fieldMeta = useFieldMeta(collections, selectedCollections);

  // Available fields (exclude filename only - allow reusing fields)
  const availableFields = useMemo(() => 
    Object.keys(fieldMeta.map).filter(f => f !== "filename"), 
    [fieldMeta.map]
  );

  const suggestions = useFilterSuggestions(
    stage, 
    input, 
    fieldMeta, 
    draft, 
    valueOptions, 
    selectedValues, 
    availableFields
  );

  // Field type checks using modern syntax
  const fieldType = draft.field ? (fieldMeta.map[draft.field] ?? "string") : "string";
  const isArrayField = fieldType === "array";
  const isDatetimeField = fieldType === "datetime";
  // Multi-select only for array operators that accept multiple values
  const shouldUseMultiSelection = isArrayField && [
    'array_contains_all',
    'array_contains_any',
    'includes',
    'does not include',
    'in',
    'not in',
  ].includes(draft.op ?? "");

  // Event handlers using modern patterns (no setTimeout)
  const commitFilter = useCallback(() => {
    const hasValue = shouldUseMultiSelection 
      ? selectedValues.length > 0
      : draft.value?.trim();
    
    if (!draft.field || !draft.op || !hasValue) return;

    const convertedValue = shouldUseMultiSelection
      ? selectedValues
      : convertValueToType(draft.value!, fieldType, draft.op, fieldMeta.arrayTypeMap, draft.field);
    
    const newFilter: Filter = {
      field: draft.field, 
      operator: draft.op as Filter['operator'], 
      value: convertedValue,
      ...(filters.length > 0 && { logicalOperator: "OR" as const }),
    };
    
    setFilters([...filters, newFilter]);
    resetToField();
  }, [shouldUseMultiSelection, selectedValues, draft, fieldType, fieldMeta.arrayTypeMap, filters, setFilters, resetToField]);

  const updateFilterLogicalOperator = useCallback((index: number, logicalOperator: "AND" | "OR") => {
    const updatedFilters = filters.map((filter, i) => 
      i === index ? { ...filter, logicalOperator } : filter
    );
    setFilters(updatedFilters);
  }, [filters, setFilters]);

  const removeFilter = useCallback((index: number) => {
    setFilters(filters.filter((_, i) => i !== index));
  }, [filters, setFilters]);

  const removeSelectedValue = useCallback((valueToRemove: string) => {
    updateState({ selectedValues: selectedValues.filter(v => v !== valueToRemove) });
  }, [selectedValues, updateState]);

  const chooseSuggestion = useCallback((idx: number) => {
    const choice = suggestions[idx];
    if (!choice) return;
    
    switch (stage) {
      case "field": {
        if (isLoadingCollections || collectionsError) return;
        updateState({ 
          draft: { field: choice }, 
          selectedValues: [], 
          stage: "op", 
          input: "",
          show: true,
          activeIdx: 0 
        });
        // Focus input for next stage
        requestAnimationFrame(() => inputRef.current?.focus());
        break;
      }
      case "op": {
        updateState({ 
          draft: { ...draft, op: choice }, 
          selectedValues: [], 
          stage: "value", 
          input: "",
        });
        // Auto-show for datetime fields
        if (fieldType === "datetime") {
          updateState({ show: true, activeIdx: 0, input: "" });
        }
        break;
      }
      case "value": {
        if (shouldUseMultiSelection) {
          updateState({ 
            selectedValues: [...selectedValues, choice], 
            input: "", 
            activeIdx: 0, 
            show: true 
          });
          requestAnimationFrame(() => inputRef.current?.focus());
        } else {
          updateState({ 
            draft: { ...draft, value: choice }, 
            input: choice, 
            show: false 
          });
          // Auto-commit for single selection
          requestAnimationFrame(() => {
            const convertedValue = convertValueToType(choice, fieldType, draft.op, fieldMeta.arrayTypeMap, draft.field);
            setFilters([...filters, { 
              field: draft.field!, 
              operator: draft.op as Filter['operator'], 
              value: convertedValue,
              ...(filters.length > 0 && { logicalOperator: "OR" as const }),
            }]);
            resetToField();
          });
        }
        break;
      }
    }
  }, [suggestions, stage, isLoadingCollections, collectionsError, updateState, draft, fieldType, shouldUseMultiSelection, selectedValues, inputRef, fieldMeta.arrayTypeMap, filters, setFilters, resetToField]);

  // Keyboard navigation using modern patterns
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (show && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      e.preventDefault();
      const next = e.key === "ArrowDown" ? activeIdx + 1 : activeIdx - 1;
      updateState({ activeIdx: (next + suggestions.length) % suggestions.length });
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (suggestions.length > 0) {
        chooseSuggestion(activeIdx);
      } else if (stage === "value" && shouldUseMultiSelection && selectedValues.length > 0) {
        commitFilter();
      } else if (stage === "value" && input.trim()) {
        commitFilter();
      }
    } else if (e.key === "Escape") {
      updateState({ show: false });
    }
  }, [show, activeIdx, suggestions.length, updateState, chooseSuggestion, stage, shouldUseMultiSelection, selectedValues.length, commitFilter, input]);

  // Get placeholder text based on current state
  const getPlaceholderText = useCallback(() => {
    if (stage === "field") {
      if (isLoadingCollections) return "Loading metadata fields...";
      if (collectionsError) return "Error loading collections";
      if (selectedCollections.length === 0) return "Select collections first to see metadata fields...";
      if (availableFields.length === 0) return "No metadata fields available";
      return "Type to search fields or press ↑↓ to browse...";
    }
    if (stage === "op") return "Press Enter to see operators or use ↑↓ keys";
    if (isDatetimeField) return "Use the datetime picker above to select date and time";
    if (valuesLoading) return "Loading values...";
    if (valueOptions.length === 0) return "No values found";
    if (shouldUseMultiSelection) {
      return selectedValues.length > 0 
        ? "Select more values or press Enter to finish..." 
        : "Select multiple values for this operator...";
    }
    return "Enter or select value";
  }, [stage, isLoadingCollections, collectionsError, selectedCollections.length, availableFields.length, isDatetimeField, valuesLoading, valueOptions.length, shouldUseMultiSelection, selectedValues.length]);

  return (
    <>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
      <Block style={STYLES.container} data-testid="filter-bar">
        <Flex align="center" gap="density-sm" style={STYLES.flex}>
        <FilterIcon />
        <FilterChips 
          filters={filters} 
          onRemoveFilter={removeFilter}
          onUpdateLogicalOperator={updateFilterLogicalOperator}
        />
        <DraftChip draft={draft} stage={stage} shouldUseMultiSelection={shouldUseMultiSelection} selectedValues={selectedValues} />
        <MultiSelectChips selectedValues={selectedValues} onRemove={removeSelectedValue} shouldShow={shouldUseMultiSelection} />
        <CommitButton 
          shouldShow={shouldUseMultiSelection && selectedValues.length > 0} 
          count={selectedValues.length}
          onCommit={commitFilter}
        />
        <FilterInput
          ref={inputRef}
          value={input}
          placeholder={getPlaceholderText()}
          onChange={(value) => {
            updateState({ input: value });
            if (stage === "value") {
              updateState({ draft: { ...draft, value } });
            }
            if (stage === "field" && !isLoadingCollections && !collectionsError) {
              updateState({ show: true, activeIdx: 0 });
            } else if (stage === "value" && valueOptions.length > 0 && !isDatetimeField) {
              updateState({ show: true, activeIdx: 0 });
            }
          }}
          onKeyDown={handleKeyDown}
          onFocus={() => updateState({ activeIdx: 0 })}
          onBlur={() => {
            if (stage === "value" && !shouldUseMultiSelection && !isDatetimeField) {
              commitFilter();
            }
          }}
          readOnly={stage === "value" && isDatetimeField}
          show={show}
          onShowChange={(show) => updateState({ show })}
          suggestions={suggestions}
          activeIdx={activeIdx}
          onChooseSuggestion={chooseSuggestion}
          stage={stage}
          fieldMeta={fieldMeta}
          isLoadingCollections={isLoadingCollections}
          collectionsError={collectionsError}
          valuesLoading={valuesLoading}
          valueOptions={valueOptions}
          selectedCollections={selectedCollections}
          shouldUseMultiSelection={shouldUseMultiSelection}
          isDatetimeField={isDatetimeField}
          onCommitFilter={commitFilter}
        />
        </Flex>
      </Block>
    </>
  );
}

// Sub-components using modern React patterns
const FilterIcon: React.FC = () => (
  <Flex align="center" gap="density-xs">
    <svg 
      style={{ width: '12px', height: '12px', color: 'var(--text-color-subtle)' }} 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="2" 
      viewBox="0 0 24 24"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c2.755 0 5.455.232 8.083.678.533.09.917.556.917 1.096v1.044a2.25 2.25 0 01-.659 1.591l-5.432 5.432a2.25 2.25 0 00-.659 1.591v2.927a2.25 2.25 0 01-1.244 2.013L9.75 21v-6.568a2.25 2.25 0 00-.659-1.591L3.659 7.409A2.25 2.25 0 013 5.818V4.774c0-.54.384-1.006.917-1.096A48.32 48.32 0 0112 3z" />
    </svg>
    <Text kind="label/regular/md" style={{ color: 'var(--text-color-subtle)' }}>
      Filters:
    </Text>
  </Flex>
);

interface FilterChipsProps {
  filters: Filter[];
  onRemoveFilter: (index: number) => void;
  onUpdateLogicalOperator: (index: number, operator: "AND" | "OR") => void;
}

const FilterChips: React.FC<FilterChipsProps> = ({ filters, onRemoveFilter, onUpdateLogicalOperator }) => (
  <>
    {filters.map((filter, i) => (
      <React.Fragment key={`${filter.field}-${i}`}>
        {i > 0 && (
          <LogicalOperatorButtons 
            operator={filter.logicalOperator}
            onChange={(op) => onUpdateLogicalOperator(i, op)}
          />
        )}
        <FilterChip filter={filter} onRemove={() => onRemoveFilter(i)} />
      </React.Fragment>
    ))}
  </>
);

interface DraftChipProps {
  draft: FilterState['draft'];
  stage: FilterState['stage'];
  shouldUseMultiSelection: boolean;
  selectedValues: string[];
}

const DraftChip: React.FC<DraftChipProps> = ({ draft, stage, shouldUseMultiSelection, selectedValues }) => {
  if (!draft.field) return null;
  
  return (
    <Tag color="gray" kind="outline" density="compact" style={{ opacity: 0.7 }} readOnly>
      <Flex align="center" gap="density-xs">
        <Text kind="body/bold/xs">{draft.field}</Text>
        {draft.op && <Text kind="body/regular/xs" style={{ opacity: 0.8 }}>{draft.op}</Text>}
        {stage === "value" && shouldUseMultiSelection && selectedValues.length > 0 && (
          <Text kind="body/regular/xs" style={{ opacity: 0.6 }}>
            [{selectedValues.length}]
          </Text>
        )}
        {stage === "value" && !shouldUseMultiSelection && (
          <Text kind="body/regular/xs" style={{ opacity: 0.6 }}>...</Text>
        )}
      </Flex>
    </Tag>
  );
};

interface MultiSelectChipsProps {
  selectedValues: string[];
  onRemove: (value: string) => void;
  shouldShow: boolean;
}

const MultiSelectChips: React.FC<MultiSelectChipsProps> = ({ selectedValues, onRemove, shouldShow }) => {
  if (!shouldShow) return null;
  
  return (
    <>
      {selectedValues.map((value) => (
        <Tag
          key={value}
          color="blue"
          kind="outline"
          density="compact"
          onClick={() => onRemove(value)}
          style={{ cursor: 'pointer' }}
        >
          <Flex align="center" gap="density-xs">
            <Text kind="body/regular/xs">"{value}"</Text>
            <svg 
              style={{ width: '10px', height: '10px' }} 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </Flex>
        </Tag>
      ))}
    </>
  );
};

interface CommitButtonProps {
  shouldShow: boolean;
  count: number;
  onCommit: () => void;
}

const CommitButton: React.FC<CommitButtonProps> = ({ shouldShow, count, onCommit }) => {
  if (!shouldShow) return null;
  
  return (
    <Button
      kind="secondary"
      color="brand"
      size="small"
      onClick={onCommit}
      style={{ fontSize: '11px', padding: '2px 8px' }}
    >
      Add Filter ({count})
    </Button>
  );
};

interface LogicalOperatorButtonsProps {
  operator?: "AND" | "OR";
  onChange: (operator: "AND" | "OR") => void;
}

const LogicalOperatorButtons: React.FC<LogicalOperatorButtonsProps> = ({ operator, onChange }) => (
  <Flex align="center" gap="density-xs">
    <Button
      kind={operator === "AND" ? "primary" : "tertiary"}
      color={operator === "AND" ? "brand" : "neutral"}
      size="small"
      onClick={() => onChange("AND")}
      style={{
        fontSize: '10px',
        padding: '2px 6px',
        minWidth: '32px',
        borderRadius: '4px 0 0 4px'
      }}
    >
      AND
    </Button>
    <Button
      kind={operator === "OR" ? "primary" : "tertiary"}
      color={operator === "OR" ? "brand" : "neutral"}
      size="small"
      onClick={() => onChange("OR")}
      style={{
        fontSize: '10px',
        padding: '2px 6px',
        minWidth: '32px',
        borderRadius: '0 4px 4px 0'
      }}
    >
      OR
    </Button>
  </Flex>
);

interface FilterChipProps {
  filter: Filter;
  onRemove: () => void;
}

const FilterChip: React.FC<FilterChipProps> = ({ filter, onRemove }) => (
  <Tag color="green" kind="outline" density="compact" onClick={onRemove}>
    <Flex align="center" gap="density-xs">
      <Text kind="body/bold/xs">{filter.field}</Text>
      <Text kind="body/regular/xs" style={{ opacity: 0.8 }}>{filter.operator}</Text>
      <Text kind="body/regular/xs">
        {formatFilterValue(filter.value)}
      </Text>
      <svg 
        style={{ width: '12px', height: '12px' }} 
        fill="none" 
        stroke="currentColor" 
        strokeWidth="2" 
        viewBox="0 0 24 24"
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    </Flex>
  </Tag>
);

interface FilterInputProps {
  value: string;
  placeholder: string;
  onChange: (value: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onFocus: () => void;
  onBlur: () => void;
  readOnly: boolean;
  show: boolean;
  onShowChange: (show: boolean) => void;
  suggestions: string[];
  activeIdx: number;
  onChooseSuggestion: (idx: number) => void;
  stage: FilterState['stage'];
  fieldMeta: FieldMeta;
  isLoadingCollections: boolean;
  collectionsError: any;
  valuesLoading: boolean;
  valueOptions: string[];
  selectedCollections: string[];
  shouldUseMultiSelection: boolean;
  isDatetimeField: boolean;
  onCommitFilter: () => void;
}

const FilterInput = React.forwardRef<HTMLInputElement, FilterInputProps>(({
  value, placeholder, onChange, onKeyDown, onFocus, onBlur, readOnly,
  show, onShowChange, suggestions, activeIdx, onChooseSuggestion,
  stage, fieldMeta, isLoadingCollections, collectionsError, valuesLoading,
  valueOptions, selectedCollections, shouldUseMultiSelection, isDatetimeField,
  onCommitFilter
}, ref) => (
  <Popover
    open={show}
    onOpenChange={onShowChange}
    side="bottom"
    align="start"
    slotContent={
      <Block style={{ 
        maxHeight: stage === "value" && isDatetimeField ? 'auto' : '160px', 
        overflowY: stage === "value" && isDatetimeField ? 'visible' : 'auto',
        padding: stage === "value" && isDatetimeField ? '12px' : '0'
      }}>
        {stage === "value" && isDatetimeField ? (
          <DatetimeInput value={value} onChange={onChange} onCommit={onCommitFilter} />
        ) : stage === "field" && isLoadingCollections ? (
          <LoadingState message="Loading metadata fields..." />
        ) : stage === "field" && collectionsError ? (
          <ErrorState message={`Error loading collections: ${collectionsError.message}`} />
        ) : stage === "value" && valuesLoading ? (
          <LoadingState message="Loading available values..." />
        ) : suggestions.length > 0 ? (
          <SuggestionsList
            suggestions={suggestions}
            activeIdx={activeIdx}
            onChoose={onChooseSuggestion}
            stage={stage}
            fieldMeta={fieldMeta}
          />
        ) : (
          <EmptyState 
            stage={stage}
            selectedCollections={selectedCollections}
            valueOptions={valueOptions}
            shouldUseMultiSelection={shouldUseMultiSelection}
            suggestions={suggestions}
          />
        )}
      </Block>
    }
  >
    <Block style={{ position: 'relative', flex: 1, minWidth: '120px' }}>
      <TextInput
        ref={ref}
        value={value}
        onChange={(e) => onChange((e.target as HTMLInputElement).value)}
        onBlur={onBlur}
        onKeyDown={onKeyDown}
        onFocus={onFocus}
        placeholder={placeholder}
        readOnly={readOnly}
        style={{
          width: '100%',
          border: 'none',
          backgroundColor: 'transparent',
          color: 'var(--text-color-primary)',
          fontSize: '14px',
          lineHeight: '20px',
          padding: '0',
          outline: 'none',
          cursor: readOnly ? 'pointer' : 'text'
        }}
      />
    </Block>
  </Popover>
));

// Additional helper components
const DatetimeInput: React.FC<{ value: string; onChange: (value: string) => void; onCommit: () => void }> = ({ value, onChange, onCommit }) => (
  <Block style={{ 
    padding: '8px',
    backgroundColor: 'var(--background-color-surface-raised)',
    border: '1px solid var(--border-color-base)',
    borderRadius: '6px'
  }}>
    <input
      type="datetime-local"
      value={value}
      onChange={(e) => {
        const val = e.target.value;
        onChange(val && val.length === 16 ? `${val}:00` : val);
      }}
      onBlur={() => value.trim() && onCommit()}
      onKeyDown={(e) => {
        if (e.key === "Enter" && value.trim()) {
          e.preventDefault();
          onCommit();
        }
      }}
      style={{
        width: '100%',
        padding: '8px 12px',
        border: 'none',
        backgroundColor: 'transparent',
        color: 'var(--text-color-primary)',
        fontSize: '14px',
        outline: 'none',
        fontFamily: 'inherit'
      }}
      autoFocus
    />
  </Block>
);

const LoadingState: React.FC<{ message: string }> = ({ message }) => (
  <Flex align="center" justify="center" gap="density-xs" style={{ padding: '12px' }}>
    <svg 
      style={{ 
        width: '16px', 
        height: '16px', 
        color: 'var(--color-brand-primary)', 
        animation: 'spin 1s linear infinite' 
      }} 
      fill="none" 
      viewBox="0 0 24 24"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" opacity="0.25" />
      <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" opacity="0.75" />
    </svg>
    <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
      {message}
    </Text>
  </Flex>
);

const ErrorState: React.FC<{ message: string }> = ({ message }) => (
  <Text kind="body/regular/sm" style={{ padding: '8px', color: 'var(--text-color-error)' }}>
    {message}
  </Text>
);

interface SuggestionsListProps {
  suggestions: string[];
  activeIdx: number;
  onChoose: (idx: number) => void;
  stage: FilterState['stage'];
  fieldMeta: FieldMeta;
}

const SuggestionsList: React.FC<SuggestionsListProps> = ({ suggestions, activeIdx, onChoose, stage, fieldMeta }) => (
  <>
    {suggestions.map((suggestion, idx) => (
      <Button
        key={suggestion}
        kind={idx === activeIdx ? "primary" : "tertiary"}
        color={idx === activeIdx ? "brand" : "neutral"}
        size="small"
        onClick={(e) => {
          e.stopPropagation();
          onChoose(idx);
        }}
        onMouseEnter={() => {}} // Could update activeIdx here if needed
        style={{
          width: '100%',
          justifyContent: stage === "field" ? 'space-between' : 'flex-start',
          fontFamily: stage === "op" ? 'monospace' : 'inherit',
          margin: '2px 0'
        }}
      >
        {stage === "field" && (
          <Flex align="center" gap="density-xs">
            <svg 
              style={{ width: '12px', height: '12px', color: 'var(--text-color-subtle)' }} 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <Text kind="body/regular/sm">{suggestion}</Text>
          </Flex>
        )}
        {stage === "field" && (
          <Text kind="label/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
            {fieldMeta.map[suggestion]}
          </Text>
        )}
        {stage === "op" && suggestion}
        {stage === "value" && `"${suggestion}"`}
      </Button>
    ))}
  </>
);

interface EmptyStateProps {
  stage: FilterState['stage'];
  selectedCollections: string[];
  valueOptions: string[];
  shouldUseMultiSelection: boolean;
  suggestions: string[];
}

const EmptyState: React.FC<EmptyStateProps> = ({ stage, selectedCollections, valueOptions, shouldUseMultiSelection, suggestions }) => {
  const getMessage = () => {
    if (stage === "field" && selectedCollections.length === 0) return "Select collections to see metadata fields";
    if (stage === "field") return "No metadata fields available";
    if (stage === "value" && valueOptions.length === 0) return "No values found";
    if (stage === "value" && shouldUseMultiSelection && suggestions.length === 0) return "All available values selected. Click 'Add Filter' or press Enter to commit.";
    return "No suggestions available";
  };

  return (
    <Text kind="body/regular/sm" style={{ padding: '8px', color: 'var(--text-color-subtle)' }}>
      {getMessage()}
    </Text>
  );
};