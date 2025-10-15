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

import { useNavigate } from "react-router-dom";
import { useNewCollectionStore } from "../../store/useNewCollectionStore";
import { useSubmitNewCollection } from "../../hooks/useSubmitNewCollection";
import { useCollections } from "../../api/useCollectionsApi";
import type { Collection } from "../../types/collections";
import { useState, useEffect, useRef, useMemo } from "react";
import { 
  Button, 
  Flex, 
  Block, 
  Notification,
  Spinner
} from "@kui/react";

// Custom hook for responsive screen detection
const useResponsiveScreen = () => {
  const [screenSize, setScreenSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 1024,
    height: typeof window !== 'undefined' ? window.innerHeight : 768
  });

  useEffect(() => {
    const handleResize = () => {
      setScreenSize({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const isMobile = screenSize.width < 768;
  const isTablet = screenSize.width >= 768 && screenSize.width < 1024;
  const isDesktop = screenSize.width >= 1024;

  return { screenSize, isMobile, isTablet, isDesktop };
};

export default function NewCollectionButtons() {
  const navigate = useNavigate();
  const { collectionName, collectionNameTouched, metadataSchema, fileMetadata, selectedFiles, isLoading, hasInvalidFiles, setError } =
    useNewCollectionStore();

  const { submit } = useSubmitNewCollection();
  const { data: existing = [] } = useCollections();
  const { isMobile, isTablet } = useResponsiveScreen();

  const [nameError, setNameError] = useState<string | null>(null);
  const isSubmittingRef = useRef(false);

  // Validate collection name on every change
  useEffect(() => {
    // Skip validation if we're in the middle of submitting
    if (isSubmittingRef.current) {
      return;
    }

    const trimmed = collectionName.trim();
    if (!trimmed) {
      setNameError(null);
      return;
    }

    const valid = /^[_a-zA-Z][_a-zA-Z0-9_]*$/.test(trimmed);
    if (!valid) {
      setNameError(
        "Name must start with a letter/underscore and contain only alphanumerics and underscores"
      );
      return;
    }

    const dup = existing.some((c: Collection) => c.collection_name === trimmed);
    if (dup) {
      setNameError("A collection with this name already exists");
      return;
    }

    setNameError(null);
  }, [collectionName, existing]);

  // Reset submission flag when loading state changes
  useEffect(() => {
    if (!isLoading) {
      isSubmittingRef.current = false;
    }
  }, [isLoading]);

  const hasMissingRequired = selectedFiles.some((file) =>
    metadataSchema.some((field) => {
      if (field.optional) return false; // Skip optional fields
      
      const value = fileMetadata[file.name]?.[field.name];
      
      // For boolean fields, both "true" and "false" are valid values
      if (field.type === "boolean") {
        if (typeof value === "boolean") return false; // Valid boolean value
        if (typeof value !== "string") return true; // Invalid type for boolean field
        const lowerValue = value.toLowerCase().trim();
        const validBooleanValues = ["true", "false", "1", "0", "yes", "no", "on", "off"];
        return !lowerValue || !validBooleanValues.includes(lowerValue);
      }
      
      // For string fields, check if value is empty
      if (field.type === "string" || field.type === "datetime") {
        return !value || (typeof value === "string" && !value.trim());
      }
      
      // For numeric fields, check if null/undefined  
      if (field.type === "integer" || field.type === "float" || field.type === "number") {
        return value === null || value === undefined;
      }
      
      // For arrays, check if empty
      if (field.type === "array") {
        return !Array.isArray(value) || value.length === 0;
      }
      
      return false;
    })
  );

  const handleSubmit = () => {
    if (nameError) {
      setError(nameError);
      return;
    }
    
    // Mark that we're starting submission to skip validation
    isSubmittingRef.current = true;
    
    // Clear any previous errors before submitting
    setError(null);
    submit();
  };

  // Generate helpful validation message explaining why button is disabled
  const getValidationMessage = () => {
    if (!collectionName.trim() && collectionNameTouched) {
      return "Please enter a collection name";
    }
    
    if (hasInvalidFiles) {
      return "Please remove or fix invalid files before creating the collection";
    }
    
    if (hasMissingRequired) {
      // Find which files have missing required fields
      const filesWithMissing = selectedFiles.filter((file) =>
        metadataSchema.some((field) => {
          if (field.optional) return false;
          const value = fileMetadata[file.name]?.[field.name];
          
          if (field.type === "boolean") {
            if (typeof value === "boolean") return false; // Valid boolean value
            if (typeof value !== "string") return true; // Invalid type for boolean field
            const lowerValue = value.toLowerCase().trim();
            const validBooleanValues = ["true", "false", "1", "0", "yes", "no", "on", "off"];
            return !lowerValue || !validBooleanValues.includes(lowerValue);
          }
          
          // For string fields, check if value is empty
          if (field.type === "string" || field.type === "datetime") {
            return !value || (typeof value === "string" && !value.trim());
          }
          
          // For numeric fields, check if null/undefined  
          if (field.type === "integer" || field.type === "float" || field.type === "number") {
            return value === null || value === undefined;
          }
          
          // For arrays, check if empty
          if (field.type === "array") {
            return !Array.isArray(value) || value.length === 0;
          }
          
          return false;
        })
      );
      
      if (filesWithMissing.length === 1) {
        return `Please fill in required fields for ${filesWithMissing[0].name}`;
      } else if (filesWithMissing.length > 1) {
        return `Please fill in required fields for ${filesWithMissing.length} files`;
      }
      return "Please fill in all required fields";
    }
    
    return null;
  };

  const validationMessage = getValidationMessage();

  // Responsive styles for the button bar
  const buttonBarStyles = useMemo(() => {
    const baseStyles = {
      backgroundColor: 'var(--background-color-surface-base)',
      borderTop: '1px solid var(--border-color-subtle)',
    };

    if (isMobile) {
      return {
        ...baseStyles,
        position: 'sticky' as const,  // Sticky positioning for mobile to work with keyboards
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 100,  // Higher z-index for mobile to ensure visibility above content
        boxShadow: '0 -2px 8px rgba(0, 0, 0, 0.1)',  // Add shadow for visual separation
        // Add safe-area-inset for devices with notches
        paddingBottom: 'env(safe-area-inset-bottom, 0px)'
      };
    }

    if (isTablet) {
      return {
        ...baseStyles,
        position: 'fixed' as const,
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 50,  // Medium z-index for tablets
        boxShadow: '0 -1px 4px rgba(0, 0, 0, 0.05)'
      };
    }

    // Desktop styles
    return {
      ...baseStyles,
      position: 'fixed' as const,
      bottom: 0,
      left: 0,
      right: 0,
      zIndex: 10  // Lower z-index for desktop
    };
  }, [isMobile, isTablet]);

  return (
    <>
      {/* Error notification - positioned normally */}
      {(nameError || validationMessage) && (
        <Block padding="4" data-testid="new-collection-buttons">
          <Notification
            status="error"
            slotHeading="Validation Error"
            slotSubheading={nameError || validationMessage}
            data-testid="error-message"
          />
        </Block>
      )}
      
      {/* Spacer to prevent content from being hidden behind the button bar */}
      <Block style={{ height: isMobile ? '80px' : '72px' }} />
      
      {/* Responsive bottom button bar */}
      <Block
        style={buttonBarStyles}
        padding={isMobile ? "3" : "4"}
        data-testid="button-container"
      >
        <Flex 
          gap={isMobile ? "2" : "3"} 
          justify="end" 
          align="center"
          direction={isMobile ? "col" : "row"}
        >
          <Button
            kind="secondary"
            onClick={() => navigate("/")}
            data-testid="cancel-button"
            size={isMobile ? "medium" : "medium"}
            style={{ width: isMobile ? '100%' : 'auto' }}
          >
            Cancel
          </Button>
          <Button
            color="brand"
            kind={isLoading ? "tertiary" : "primary"}
            disabled={!collectionName.trim() || hasMissingRequired || hasInvalidFiles || isLoading || !!nameError}
            onClick={handleSubmit}
            data-testid="create-button"
            size={isMobile ? "medium" : "medium"}
            style={{ width: isMobile ? '100%' : 'auto' }}
          >
            {isLoading ? <Spinner description="" size="small" /> : "Create Collection"}
          </Button>
        </Flex>
      </Block>
    </>
  );
}
