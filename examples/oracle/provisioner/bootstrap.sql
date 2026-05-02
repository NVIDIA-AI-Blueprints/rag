-- =============================================================================
-- bootstrap.sql — Bootstrap an Oracle Autonomous AI Database (23ai/26ai)
-- for use as the vector store backend of the NVIDIA RAG Blueprint.
-- =============================================================================
-- Run as the ADB ADMIN user (e.g. via Oracle SQL Developer / Database Actions
-- or sqlplus). Creates or updates a dedicated RAG_APP schema with the
-- privileges the OracleVDB plugin needs (DDL on its own schema + vector +
-- Oracle Text indexing).
--
-- Usage:
--   sqlplus ADMIN/<ADMIN_PASSWORD>@<TNS_ALIAS> @bootstrap.sql
--   (substitution variable &rag_app_password will be prompted)
--
-- Vector tables and indexes themselves (one per RAG collection) are created
-- dynamically by OracleVDB at runtime when /v1/collections is first hit.
-- This script only bootstraps the application user + privileges.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Create RAG_APP user if needed
-- -----------------------------------------------------------------------------
DECLARE
    user_exists NUMBER;
BEGIN
    SELECT COUNT(*) INTO user_exists FROM dba_users WHERE username = 'RAG_APP';
    IF user_exists = 0 THEN
        EXECUTE IMMEDIATE 'CREATE USER RAG_APP IDENTIFIED BY "&rag_app_password" DEFAULT TABLESPACE DATA QUOTA UNLIMITED ON DATA';
    ELSE
        -- Non-destructive retry behavior: never drop customer data. Re-apply
        -- password + grants below so reruns repair configuration without
        -- removing existing collections.
        EXECUTE IMMEDIATE 'ALTER USER RAG_APP IDENTIFIED BY "&rag_app_password"';
    END IF;
END;
/

-- -----------------------------------------------------------------------------
-- 2. Grants — privileges required by python-oracledb + OracleVDB operations
-- -----------------------------------------------------------------------------
-- Session + basic DML/DDL on own schema
GRANT CONNECT, RESOURCE TO RAG_APP;

-- Explicit grants (RESOURCE role doesn't include some needed privileges in ADB)
GRANT CREATE SESSION  TO RAG_APP;
GRANT CREATE TABLE    TO RAG_APP;
GRANT CREATE VIEW     TO RAG_APP;
GRANT CREATE SEQUENCE TO RAG_APP;
GRANT CREATE PROCEDURE TO RAG_APP;

-- Vector-index privileges (native VECTOR type + VECTOR INDEX DDL)
GRANT CREATE MINING MODEL TO RAG_APP;

-- Oracle Text — required for hybrid (vector + keyword) retrieval
GRANT EXECUTE ON CTXSYS.CTX_DDL TO RAG_APP;
GRANT EXECUTE ON CTXSYS.CTX_QUERY TO RAG_APP;
GRANT CTXAPP TO RAG_APP;
GRANT CREATE ANY INDEX TO RAG_APP;

-- JSON / BLOB / CLOB handling (content_metadata + source columns)
GRANT SELECT ANY DICTIONARY TO RAG_APP;

-- -----------------------------------------------------------------------------
-- 3. Sanity check
-- -----------------------------------------------------------------------------
SELECT username, account_status, default_tablespace
  FROM dba_users
 WHERE username = 'RAG_APP';

-- Confirm vector data type is available
SELECT 'VECTOR datatype present' AS status
  FROM dual
 WHERE EXISTS (SELECT 1 FROM v$version
                WHERE banner LIKE 'Oracle Database 23%'
                   OR banner LIKE 'Oracle Database 26%');

PROMPT ========================================================================
PROMPT  RAG_APP user created. Test connection from a shell:
PROMPT
PROMPT    export TNS_ADMIN=<wallet-dir>
PROMPT    python -c "import oracledb; \
PROMPT      conn = oracledb.connect(user='RAG_APP', password='<pw>', dsn='<tns_alias>'); \
PROMPT      print(conn.version)"
PROMPT ========================================================================
