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

import { 
  Button, 
  Flex, 
  Spinner 
} from "@kui/react";

interface DrawerActionsProps {
  onDelete: () => void;
  onAddSource: () => void;
  onCloseUploader: () => void;
  isDeleting?: boolean;
  showUploader?: boolean;
}

const DeleteIcon = () => (
  <svg 
    style={{ width: '16px', height: '16px' }}
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
  </svg>
);

const AddIcon = () => (
  <svg 
    style={{ width: '16px', height: '16px' }}
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
  </svg>
);

const CloseIcon = () => (
  <svg 
    style={{ width: '16px', height: '16px' }}
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

export const DrawerActions = ({ 
  onDelete, 
  onAddSource, 
  onCloseUploader,
  isDeleting = false,
  showUploader = false
}: DrawerActionsProps) => (
  <Flex gap="3" justify="stretch" style={{ width: '100%' }}>
    <Button
      kind="secondary"
      color="danger"
      onClick={onDelete}
      disabled={isDeleting}
    >
      {isDeleting ? (
        <>
          <Spinner size="small" description="" />
          Deleting...
        </>
      ) : (
        <>
          <DeleteIcon />
          Delete Collection
        </>
      )}
    </Button>
    
    <Button
      color={showUploader ? "neutral" : "brand"}
      onClick={showUploader ? onCloseUploader : onAddSource}
    >
      {showUploader ? <CloseIcon /> : <AddIcon />}
      {showUploader ? "Close Uploader" : "Add Source to Collection"}
    </Button>
  </Flex>
); 