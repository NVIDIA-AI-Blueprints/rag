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

import { NextRequest, NextResponse } from "next/server";
import {
  createErrorResponse,
  validateRequiredFields,
} from "../utils/api-utils";
import { API_CONFIG } from "@/app/config/api";

// POST /collection
export async function POST(request: NextRequest) {
  try {
    const requestData = await request.json();

    validateRequiredFields(requestData, ["collection_name", "embedding_dimension"]);

    const payload = {
      vdb_endpoint: API_CONFIG.VDB.VDB_ENDPOINT,
      collection_name: requestData.collection_name,
      embedding_dimension: requestData.embedding_dimension,
      metadata_schema: requestData.metadata_schema ?? [],
    };

    const url = `${API_CONFIG.VDB.BASE_URL}/collection`;

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Failed to create collection: ${response.statusText}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return createErrorResponse(error);
  }
}