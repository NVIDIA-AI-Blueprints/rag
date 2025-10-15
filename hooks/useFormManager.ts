import { useState, useCallback, useMemo } from "react";

interface ValidationRule {
  min?: number;
  max?: number;
  required?: boolean;
  pattern?: RegExp;
  validator?: (value: any) => boolean | string;
}

interface FormField {
  value: any;
  stringValue: string;
  isValid: boolean;
  error?: string;
  hasChanged: boolean;
  rules?: ValidationRule;
}

interface FormManagerOptions {
  onFieldChange?: (name: string, value: any) => void;
  validateOnChange?: boolean;
}

export const useFormManager = (options: FormManagerOptions = {}) => {
  const [fields, setFields] = useState<Record<string, FormField>>({});
  const { onFieldChange, validateOnChange = true } = options;

  const registerField = useCallback((
    name: string, 
    initialValue: any, 
    rules?: ValidationRule
  ) => {
    const stringValue = Array.isArray(initialValue) 
      ? initialValue.join(", ") 
      : initialValue?.toString() || "";
    
    setFields(prev => ({
      ...prev,
      [name]: {
        value: initialValue,
        stringValue,
        isValid: true,
        hasChanged: false,
        rules,
      }
    }));
  }, []);

  const validateField = useCallback((value: any, rules?: ValidationRule): { isValid: boolean; error?: string } => {
    if (!rules) return { isValid: true };

    if (rules.required && (!value || (typeof value === 'string' && !value.trim()))) {
      return { isValid: false, error: "This field is required" };
    }

    if (rules.min && typeof value === 'number' && value < rules.min) {
      return { isValid: false, error: `Must be at least ${rules.min}` };
    }

    if (rules.max && typeof value === 'number' && value > rules.max) {
      return { isValid: false, error: `Must be at most ${rules.max}` };
    }

    if (rules.pattern && typeof value === 'string' && !rules.pattern.test(value)) {
      return { isValid: false, error: "Invalid format" };
    }

    if (rules.validator) {
      const result = rules.validator(value);
      if (typeof result === 'string') {
        return { isValid: false, error: result };
      }
      if (!result) {
        return { isValid: false, error: "Invalid value" };
      }
    }

    return { isValid: true };
  }, []);

  const updateField = useCallback((
    name: string, 
    stringValue: string, 
    parser?: (value: string) => any
  ) => {
    setFields(prev => {
      const field = prev[name];
      if (!field) return prev;

      let parsedValue = stringValue;
      let validation = { isValid: true, error: undefined };

      if (parser) {
        try {
          parsedValue = parser(stringValue);
        } catch (error) {
          validation = { isValid: false, error: "Invalid format" };
        }
      }

      if (validation.isValid && validateOnChange) {
        validation = validateField(parsedValue, field.rules);
      }

             const updatedField = {
         value: parsedValue,
         stringValue,
         isValid: validation.isValid,
         error: validation.error || undefined,
         hasChanged: true,
         rules: field.rules,
       };

      if (validation.isValid && onFieldChange) {
        onFieldChange(name, parsedValue);
      }

      return {
        ...prev,
        [name]: updatedField
      };
    });
  }, [validateField, validateOnChange, onFieldChange]);

  const syncField = useCallback((name: string, externalValue: any) => {
    setFields(prev => {
      const field = prev[name];
      if (!field) return prev;

      const stringValue = Array.isArray(externalValue) 
        ? externalValue.join(", ") 
        : externalValue?.toString() || "";

      return {
        ...prev,
        [name]: {
          ...field,
          value: externalValue,
          stringValue,
          hasChanged: false,
        }
      };
    });
  }, []);

  const getField = useCallback((name: string) => fields[name], [fields]);

  const isFieldValid = useCallback((name: string) => {
    return fields[name]?.isValid ?? true;
  }, [fields]);

  const resetField = useCallback((name: string) => {
    setFields(prev => {
      const field = prev[name];
      if (!field) return prev;

      return {
        ...prev,
        [name]: {
          ...field,
          hasChanged: false,
          error: undefined,
          isValid: true,
        }
      };
    });
  }, []);

  const resetAllFields = useCallback(() => {
    setFields(prev => {
      const reset: Record<string, FormField> = {};
      Object.keys(prev).forEach(key => {
        reset[key] = {
          ...prev[key],
          hasChanged: false,
          error: undefined,
          isValid: true,
        };
      });
      return reset;
    });
  }, []);

  const validateAllFields = useCallback(() => {
    let allValid = true;
    setFields(prev => {
      const validated: Record<string, FormField> = {};
      
      Object.keys(prev).forEach(key => {
        const field = prev[key];
        const validation = validateField(field.value, field.rules);
        
        validated[key] = {
          ...field,
          isValid: validation.isValid,
          error: validation.error,
        };

        if (!validation.isValid) {
          allValid = false;
        }
      });

      return validated;
    });

    return allValid;
  }, [validateField]);

  const formState = useMemo(() => {
    const fieldNames = Object.keys(fields);
    const hasErrors = fieldNames.some(name => !fields[name].isValid);
    const hasChanges = fieldNames.some(name => fields[name].hasChanged);
    const values = fieldNames.reduce((acc, name) => {
      acc[name] = fields[name].value;
      return acc;
    }, {} as Record<string, any>);

    return {
      isValid: !hasErrors,
      hasChanges,
      values,
      errors: fieldNames.reduce((acc, name) => {
        if (fields[name].error) {
          acc[name] = fields[name].error;
        }
        return acc;
      }, {} as Record<string, string>),
    };
  }, [fields]);

  return {
    // Field management
    registerField,
    updateField,
    syncField,
    getField,
    isFieldValid,
    
    // Form management
    resetField,
    resetAllFields,
    validateAllFields,
    
    // State
    formState,
    fields,
  };
}; 