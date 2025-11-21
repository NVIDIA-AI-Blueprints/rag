<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# User Interface for NVIDIA RAG Blueprint

After you [deploy the NVIDIA RAG Blueprint](readme.md#deploy), 
use the following procedure to start testing and experimenting in the NVIDIA RAG Blueprint User Interface (RAG UI).

> [!Important]
> The RAG UI is provided as a sample and for experimentation only. It is not intended for your production environment. 


1. Open a web browser and navigate to `http://localhost:8090` for a local deployment or `http://<workstation-ip-address>:8090` for a remote deployment. 

   The RAG UI appears.

    <p align="left">
        <img src="assets/ui-empty.png" width="750">
    </p>

2. Click **New Collection** to add a new collection of documents. The **Create New Collection** dialog appears.

    <p align="left">
        <img src="assets/ui-create-new.png" width="750">
    </p>

3. Choose some files to upload in the collection.  Wait while the files are ingested.

    > [!NOTE]
    > The UI file upload interface has a hard limit of **100 files per upload batch**. When selecting more than 100 files, only the first 100 are processed. For bulk uploads beyond this limit, use multiple upload batches or the [programmatic API](../notebooks/ingestion_api_usage.ipynb).

4. Create two collections, one named *test_collection_1* and one named *test_collection_2*.

5. For **Collections**, add the two collections that you created.

6. In **Ask a question about your documents**, submit a query related (or not) to the documents that you uploaded to the collections.  You can query a minimum of 1 and a maximum of 5 collections. You should see results similar to the following.

    <p align="left">
        <img src="assets/ui-query-response.png" width="750">
    </p>

7. (Optional) Click **Sources** to view the documents that were used to generate the answer.

8. (Optional) Click **Settings** to experiment with the settings to see the effect on generated answers.


## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Get Started](deploy-docker-self-hosted.md)
- [Notebooks](notebooks.md)
