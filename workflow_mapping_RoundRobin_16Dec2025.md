# Round Robin Lead Notifier - Workflow Mapping - 16 Dec 2025

## 📋 System Overview

This script automatically distributes leads from Google Sheets and sends email notifications to TCV.

**Trigger mechanism**: Time-driven polling (every 1 minute)

---

## 🔄 Main Workflow

```mermaid
flowchart TD
    Start([Trigger every 1 min]) --> CheckEnabled{CONFIG.ENABLED?}
    CheckEnabled -->|false| Stop([Stop])
    CheckEnabled -->|true| PollNewRows[pollNewRows]
    
    PollNewRows --> ForEachSheet[Loop through each Sheet]
    ForEachSheet --> CheckSheetExists{Sheet exists?}
    CheckSheetExists -->|No| LogError[Log error]
    CheckSheetExists -->|Yes| GetLastRow[Get lastRow from Properties]
    
    GetLastRow --> CompareRows{lastRow > lastDone?}
    CompareRows -->|No| UpdateProperty[Update Properties]
    CompareRows -->|Yes| ScanNewRows[Scan new rows]
    
    ScanNewRows --> CheckTriggerCol{Column B has data?}
    CheckTriggerCol -->|No| Skip[Skip]
    CheckTriggerCol -->|Yes| AddToQueue[Add to queue]
    
    AddToQueue --> HandleEvent[handleEvent_]
    HandleEvent --> CheckSpam{Spam filter?}
    CheckSpam -->|Yes| SkipSpam[Skip spam]
    CheckSpam -->|No| CheckDuplicate{Sent recently?}
    
    CheckDuplicate -->|Yes| SkipDup[Skip duplicate]
    CheckDuplicate -->|No| AssignLP[Assign LP round-robin]
    
    AssignLP --> BuildEmail[Build email HTML + text]
    BuildEmail --> CheckDryRun{DRY_RUN?}
    CheckDryRun -->|true| LogOnly[Log only]
    CheckDryRun -->|false| SendEmail[Send email via MailApp]
    
    SendEmail --> MarkAsSent[Save timestamp]
    MarkAsSent --> UpdateProperty
    UpdateProperty --> Stop
    
    LogError --> Stop
    Skip --> UpdateProperty
    SkipSpam --> Stop
    SkipDup --> Stop
    LogOnly --> Stop
```

---

## 🏗️ Component Architecture

```mermaid
graph LR
    subgraph "Triggers"
        T1[Time-driven<br/>every 1 min]
    end
    
    subgraph "Core Functions"
        P[pollNewRows]
        H[handleEvent_]
        S[isSpamRow_]
    end
    
    subgraph "Email Builders"
        B1[buildHtmlEmail_]
        B2[buildTextEmail_]
        R1[renderLeadRows_]
        R2[renderTableHeader_]
    end
    
    subgraph "Utilities"
        U1[letterToCol_]
        U2[isFilled]
        U3[safeStr]
        U4[escapeHtml_]
    end
    
    subgraph "Control Functions"
        C1[enableScript]
        C2[disableScript]
        C3[checkStatus]
        C4[setStartRow]
    end
    
    subgraph "Manual Testing"
        M1[processSpecificRows]
        M2[simulate]
        M3[sendTest]
    end
    
    T1 --> P
    P --> H
    H --> S
    H --> B1
    H --> B2
    B1 --> R1
    B1 --> R2
    
    H --> U1
    H --> U2
    H --> U3
    B1 --> U4
    B2 --> U4
```

---

## 📊 Data Flow

```mermaid
sequenceDiagram
    participant Timer as Google Trigger<br/>(every 1 min)
    participant Poll as pollNewRows
    participant Props as Properties Service
    participant Sheet as Google Sheets
    participant Handler as handleEvent_
    participant Spam as SpamFilter
    participant RR as Round Robin Logic
    participant Mail as MailApp
    
    Timer->>Poll: Trigger
    Poll->>Props: Get lastRow processed
    Props-->>Poll: lastRow = X
    
    Poll->>Sheet: getLastRow
    Sheet-->>Poll: currentRow = Y
    
    alt Y > X - new rows found
        Poll->>Sheet: Scan rows X+1 to Y
        Sheet-->>Poll: List of rows with data
        
        loop Each new row
            Poll->>Handler: handleEvent_ row
            Handler->>Spam: isSpamRow_?
            
            alt Spam detected
                Spam-->>Handler: true
                Handler->>Handler: Skip row
            else Clean row
                Spam-->>Handler: false
                Handler->>Props: Check seen timestamp
                
                alt Sent recently
                    Props-->>Handler: timestamp recent
                    Handler->>Handler: Skip duplicate
                else Not sent
                    Props-->>Handler: OK to send
                    Handler->>RR: Assign LP round-robin
                    RR-->>Handler: LP assigned
                    
                    Handler->>Handler: Build email content
                    Handler->>Mail: sendEmail
                    Mail-->>Handler: Email sent
                    
                    Handler->>Props: Save timestamp
                end
            end
        end
        
        Poll->>Props: Update lastRow = Y
    end
```

---

## 🎯 Key Components

### 1. Trigger Setup
**Function**: `setupTriggers()`

```javascript
setupTriggers(); // Run once to setup
```

**Actions**:
- Delete old triggers (if any)
- Create time-driven trigger running every 1 minute
- Call `initPollBaseline()` to set initial baseline

---

### 2. Polling Mechanism
**Function**: `pollNewRows()`

**Flow**:
1. Check `CONFIG.ENABLED`
2. Loop through each sheet in `CONFIG.SHEET_NAMES`
3. Compare `lastRow` (current) vs `lastDone` (from Properties)
4. If new rows exist → scan and check column B (Phone)
5. Send each row with data to `handleEvent_()`

**State Management**:
- Uses `PropertiesService` to store `lastRow:${sheetId}`
- Each sheet has its own state

---

### 3. Event Handler
**Function**: `handleEvent_(e, sourceType)`

**Validation Chain**:
```mermaid
flowchart LR
    Start[Row Input] --> V1{Sheet valid?}
    V1 -->|No| Stop1[Return]
    V1 -->|Yes| V2{Row >= MIN_ROW?}
    V2 -->|No| Stop2[Return]
    V2 -->|Yes| V3{Has data?}
    V3 -->|No| Stop3[Return]
    V3 -->|Yes| V4{Spam check?}
    V4 -->|Yes| Stop4[Skip spam]
    V4 -->|No| V5{Sent recently?}
    V5 -->|Yes| Stop5[Skip duplicate]
    V5 -->|No| Process[Process & Send]
```

**Processing Steps**:
1. **Validation**: CONFIG, sheet, row number
2. **Data extraction**: Get Name, Phone, Leadcode
3. **Spam filter**: Check keywords in columns A, B, D, E, H
4. **Duplicate check**: Check timestamp (5-minute window)
5. **LP assignment**: Round-robin from LP_POOL
6. **Email generation**: HTML + plain text
7. **Send email**: MailApp.sendEmail()
8. **State update**: Save timestamp, update LP index

---

### 4. Spam Filter
**Function**: `isSpamRow_(sh, rowNum)`

**Keywords** (47 total):
- Test/Demo: `test`, `demo`, `sample`, `thử`
- Spam/Fake: `spam`, `fake`, `giả`, `lừa đảo`
- Invalid: `xxx`, `aaa`, `null`, `invalid`
- Test names: `nguyen van a`, `tran thi b`
- Test phones: `0000000000`, `1111111111`
- Bot: `bot`, `robot`, `automated`
- Profanity:_________________________________
- Suspicious: `hack`, `virus`, `malware`

**Check columns**: A, B, D, E, H  
**Method**: Case-insensitive substring match

---

### 5. Round Robin Logic
**Location**: Inside `handleEvent_()` with Lock

```javascript
// Lock for thread-safety
const lock = LockService.getDocumentLock();
lock.waitLock(30000);

// Get current LP index
let lpIdx = parseInt(props.getProperty(keyIdx) || '0', 10);

// Assign LP and increment
const lp = pool[lpIdx % pool.length];
lpIdx++;

// Save index
props.setProperty(keyIdx, String(lpIdx));
```

**Features**:
- Thread-safe with `LockService`
- Each sheet has its own LP index
- Auto wraps when reaching end of pool

---

### 6. Email Builder
**Functions**: `buildHtmlEmail_()`, `buildTextEmail_()`

**HTML Email Structure**:
```
┌─────────────────────────────────┐
│ Header - Brand + Sheet info     │
├─────────────────────────────────┤
│ Table:                          │
│ ┌───┬────────┬────────┬────────┐│
│ │ # │ Name   │ Phone  │Leadcode││
│ ├───┼────────┼────────┼────────┤│
│ │100│John Doe│0901... │LC-123  ││
│ └───┴────────┴────────┴────────┘│
├─────────────────────────────────┤
│ LP Note - Round-robin info      │
└─────────────────────────────────┘
```

**Theme**: Customizable via `CONFIG.EMAIL`

---

## 🎮 Control Functions

### Quick Controls

```javascript
// Enable/disable script
enableScript();
disableScript();

// Check status
checkStatus();

// Set starting row
setStartRow(100);      // From row 100
setStartRow('auto');   // Automatic

// Reset to auto
resetToAuto();

// Spam filter controls
enableSpamFilter();
disableSpamFilter();

// Dry run - test mode
enableDryRun();        // Log only, no emails
disableDryRun();       // Send real emails
```

---

## 🧪 Manual Testing Functions

### 1. Process Specific Rows
```javascript
processSpecificRows([100, 101, 102]);
```

### 2. Process All Pending
```javascript
processAllPendingRows();
```

### 3. Simulate
```javascript
simulate(2);  // Simulate processing row 2
```

### 4. Send Test Email
```javascript
sendTest();  // Send test email with dummy data
```

### 5. Validate Config
```javascript
validateConfig();  // Check configuration
```

---

## 🔧 Configuration

### CONFIG Object

```javascript
CONFIG = {
  ENABLED: true,                    // Master switch
  START_FROM_ROW: null,             // Auto or specific row
  DRY_RUN: false,                   // Test mode
  
  RECIPIENTS_CC: [...],             // CC emails
  LP_POOL: [{name, email}, ...],    // LP list
  
  SHEET_NAMES: [...],               // Target sheets
  TARGET_COLUMN: 'B',               // Trigger column
  DATA_COLS: {                      // Data columns
    NAME: 'A',
    PHONE: 'B',
    LEADCODE: 'F',
    COL_D: 'D',
    COL_E: 'E',
    COL_H: 'H'
  },
  
  SPAM_FILTER: {
    ENABLED: true,
    CASE_SENSITIVE: false,
    KEYWORDS: [...],
    CHECK_COLUMNS: ['A','B','D','E','H']
  },
  
  MIN_ROW: 2,
  MAX_LINES: 50,
  
  EMAIL: {
    brand: '...',
    primary: '#C00000',
    bg: '#f7f7f9',
    text: '#222',
    border: '#e5e7eb'
  }
}
```

---

## 📝 Example Scenarios

### Scenario 1: New Lead Added

```mermaid
sequenceDiagram
    participant User
    participant Sheet as Google Sheet
    participant Timer
    participant Script
    participant LP as LP Email
    
    User->>Sheet: Add row 100 with phone
    Note over Sheet: Row 100: Name="John", Phone="0901234567"
    
    Timer->>Script: Trigger pollNewRows
    Script->>Sheet: Check lastRow
    Sheet-->>Script: lastRow = 100, lastDone = 99
    
    Script->>Sheet: Scan row 100, column B
    Sheet-->>Script: "0901234567" - has data
    
    Script->>Script: isSpamRow_? No
    Script->>Script: Sent recently? No
    Script->>Script: Assign LP - round-robin
    Script->>Script: Build email
    Script->>LP: Send email
    
    Script->>Script: Update lastDone = 100
    Script->>Script: Save timestamp
```

**Timeline**:
- T+0s: User adds row
- T+60s: Trigger fires
- T+61s: Email sent
- T+120s: Next trigger (no new rows)

---

### Scenario 2: Spam Detected

```mermaid
sequenceDiagram
    participant User
    participant Sheet
    participant Script
    
    User->>Sheet: Add row: Name="test"
    Note over Sheet: Row 101: Name="test", Phone="123"
    
    Script->>Sheet: Poll row 101
    Sheet-->>Script: Data found
    
    Script->>Script: isSpamRow_?
    Note over Script: Check "test" in keywords
    Script->>Script: Spam detected!
    
    Script->>Script: Skip row 101
    Script->>Script: No email sent
```

**Result**: Row skipped, no email sent

---

### Scenario 3: Multiple New Rows

```mermaid
flowchart LR
    A[Rows 100-105 added] --> B[pollNewRows]
    B --> C{Scan 100-105}
    
    C --> D[Row 100: Has data]
    C --> E[Row 101: Empty]
    C --> F[Row 102: Has data]
    C --> G[Row 103: Spam]
    C --> H[Row 104: Has data]
    C --> I[Row 105: Empty]
    
    D --> J[Email to LP1]
    E --> K[Skip]
    F --> L[Email to LP2]
    G --> M[Skip spam]
    H --> N[Email to LP1]
    I --> O[Skip]
```

**Result**: 3 emails sent (rows 100, 102, 104) with round-robin LP assignment

---

## 🛡️ Safety Features

### 1. Duplicate Prevention
- **Mechanism**: Timestamp in Properties
- **Window**: 5 minutes
- **Key**: `seen:${sheetId}:${rowNum}`

### 2. Spam Filter
- **Check**: 47 keywords
- **Columns**: A, B, D, E, H
- **Method**: Case-insensitive substring

### 3. Validation Chain
- Sheet exists
- Row >= MIN_ROW
- Has data in trigger column
- Not spam
- Not sent recently

### 4. Thread Safety
- **Lock**: `LockService.getDocumentLock()`
- **Timeout**: 30 seconds
- **Scope**: Per document

### 5. Error Handling
- Try-catch blocks
- Console logging
- Email error notifications

---

## 🔍 Debugging Tips

### 1. Check Status
```javascript
checkStatus();
// Output: 
// ✅ Round Robin status: ON
// 📍 Starting row: Automatic (current last row)
```

### 2. Validate Configuration
```javascript
validateConfig();
// Checks sheets, columns, LP pool, email quota
```

### 3. Enable Dry Run
```javascript
enableDryRun();
// Log only, no real emails sent
```

### 4. Test Specific Row
```javascript
simulate(100);
// Simulate processing row 100
```

### 5. Check Logs
- Execution Logs in Google Apps Script Editor
- Look for: ⚠️, ❌, ✅, 📧 icons

---

## 📊 State Management

### Properties Service Keys

| Key | Value | Purpose |
|-----|-------|---------|
| `lastRow:${sheetId}` | Integer | Last processed row |
| `lpIdx:${sheetId}` | Integer | Current LP index |
| `seen:${sheetId}:${rowNum}` | Timestamp | Email sent time |

### Lifecycle

```mermaid
stateDiagram-v2
    [*] --> NotSetup
    NotSetup --> Setup: setupTriggers()
    Setup --> Polling: Trigger fires
    Polling --> Processing: New rows found
    Processing --> Polling: Email sent
    Polling --> Polling: No new rows
    
    Setup --> Disabled: disableScript()
    Disabled --> Setup: enableScript()
```

---

## 🎯 Best Practices

### 1. Setup
```javascript
// Run once when setting up new installation
setupTriggers();
```

### 2. Production Use
```javascript
// Ensure correct configuration
CONFIG.ENABLED = true;
CONFIG.DRY_RUN = false;
CONFIG.SPAM_FILTER.ENABLED = true;
```

### 3. Testing
```javascript
// Enable dry run when testing
enableDryRun();
simulate(2);
disableDryRun();
```

### 4. Monitoring
```javascript
// Periodically check status
checkStatus();
validateConfig();
```

---

## 🚀 Quick Start Guide

### 1. Initial Setup
```javascript
// Run in Script Editor
setupTriggers();
```

### 2. Verify
```javascript
checkStatus();
validateConfig();
```

### 3. Test
```javascript
enableDryRun();
sendTest();
// Check logs
disableDryRun();
```

### 4. Production
```javascript
enableScript();
// Script auto-runs every 1 minute
```

---

## 📌 Summary

**Main Workflow**:
1. ⏰ Timer trigger every 1 minute
2. 🔍 Poll sheets to find new rows
3. ✅ Validate & filter spam
4. 🎲 Round-robin assign LP
5. 📧 Build & send email
6. 💾 Update state

**Key Features**:
- ✅ Auto polling every 1 minute
- ✅ Round-robin LP assignment
- ✅ Spam filter with 47 keywords
- ✅ Duplicate prevention (5 min window)
- ✅ Thread-safe with Lock
- ✅ Manual testing functions
- ✅ Flexible configuration



