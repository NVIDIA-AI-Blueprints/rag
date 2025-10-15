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

interface CitationVisualContentProps {
  imageData: string;
  documentType: string;
}

export const CitationVisualContent = ({ 
  imageData, 
  documentType 
}: CitationVisualContentProps) => (
  <div className="flex justify-center">
    <img
      src={`data:image/png;base64,${imageData}`}
      alt={`Citation ${documentType}`}
      className="max-w-full h-auto rounded-lg border border-neutral-700 shadow-lg"
    />
  </div>
); 