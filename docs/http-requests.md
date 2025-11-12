# Understanding HTTP Requests 🌐

## What is an HTTP Request?

HTTP requests are how your **browser talks to websites**. Every time you:
- Load a webpage
- Submit a form
- Click a link
- Upload a file

Your browser sends an HTTP request to the server.

## 📧 The Structure of an HTTP Request

### Example Request (What We Analyze)

```
POST /login.php HTTP/1.1                    ← Request Line
Host: example.com                           ← Headers
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
Content-Type: application/x-www-form-urlencoded
Content-Length: 35

username=admin&password=secret123         ← Body/Content
```

## 🧩 Parts We Extract Features From

### 1. Method (GET, POST, PUT, DELETE)
```python
# Examples:
"GET"     # Normal - just requesting data
"POST"    # Normal - sending form data
"DELETE"  # Could be suspicious depending on context
```

### 2. URL (The web address)
```python
# Normal URLs:
/login.php
/search?q=cats&page=1
/user/profile/123

/api/users/123
/admin/dashboard
```

### 3. Headers (Extra information)
```python
# User-Agent (What browser/device)
"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
"sqlmap/1.4.5"  # ← SUSPICIOUS! Attack tool!

# Content-Type
"application/x-www-form-urlencoded"  # Normal form data
"text/html"                         # Normal webpage

# Content-Length
"150"  # Size of request body in bytes
```

### 4. Body/Content (Form data, file uploads)
```python
# Normal form submission:
"username=john&password=mypass&remember=on"

# File upload:
------boundary123
Content-Disposition: form-data; name="file"; filename="photo.jpg"
Content-Type: image/jpeg

(binary image data here)
------boundary123--
```

## 🚨 Common Web Attacks (What We Detect)

### 1. SQL Injection
**What it is**: Attacker tries to run database commands through forms
```sql
-- Normal login:
SELECT * FROM users WHERE username='john' AND password='mypass'

-- SQL Injection attack:
SELECT * FROM users WHERE username='admin' OR '1'='1' -- AND password=''
-- This makes the WHERE clause always true!
```

**In HTTP request:**
```
POST /login.php HTTP/1.1
Content-Length: 50

username=admin' OR '1'='1&password=
```

### 2. Cross-Site Scripting (XSS)
**What it is**: Attacker injects JavaScript that runs in other users' browsers
```html
<!-- Normal comment -->
<p>Hello, welcome to our site!</p>

<!-- XSS attack -->
<script>alert('Hacked!'); document.cookie='stolen';</script>
```

**In HTTP request:**
```
POST /comment.php HTTP/1.1
Content-Length: 80

comment=<script>alert('Hacked!')</script>&submit=Post
```

### 3. Path Traversal (Directory Traversal)
**What it is**: Attacker tries to access files outside the web directory
```bash
# Normal file request:
/var/www/html/images/logo.png

# Path traversal attack:
/var/www/html/../../../etc/passwd
# This reads the server's password file!
```

**In HTTP request:**
```
GET /download.php?file=../../../etc/passwd HTTP/1.1
```

### 4. Command Injection
**What it is**: Attacker tries to run system commands on the server
```bash
# Normal ping command:
ping google.com

# Command injection:
ping google.com; rm -rf /  # Deletes everything!
```

**In HTTP request:**
```
POST /ping.php HTTP/1.1
Content-Length: 25

host=google.com; rm -rf /
```

## 🔍 Features We Extract

### URL Features
- Length of URL
- Number of query parameters (`?name=value&other=123`)
- Contains suspicious characters (`< > " ' ; | $`)
- Has encoded characters (`%3C` instead of `<`)

### Content Features
- Content length
- Contains suspicious patterns
- Has encoded characters

### Header Features
- User-Agent looks like attack tool?
- Content-Type is expected type?
- Host header is correct?

### Pattern Detection
- SQL keywords (`SELECT`, `UNION`, `DROP`)
- XSS patterns (`<script>`, `javascript:`)
- File system patterns (`../`, `..\\`)
- Command patterns (`;`, `|`, `` ` ``)

## 📊 Real Examples from Our Data

### Normal Request (Class 0)
```
GET /tienda1/publico/anadir.jsp?id=2&nombre=Vino+Rioja&precio=100&cantidad=1&B1=A%F1adir+al+carrito HTTP/1.1
Host: localhost:8080
User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
```

**Features extracted:**
- URL length: 85
- Query params: 5
- No suspicious chars: ✅
- Standard User-Agent: ✅

### Anomalous Request (Class 1)
```
GET /tienda1/publico/anadir.jsp?id=2'; DROP TABLE usuarios; -- &nombre=Vino+Rioja&precio=100&cantidad=1&B1=A%F1adir+al+carrito HTTP/1.1
```

**Features extracted:**
- Has SQL keywords: ❌
- Has suspicious chars: ❌
- SQL injection pattern detected: ❌

## 🎯 Why This Matters

**Before ML:**
- Security experts write rules: "Block if contains 'DROP TABLE'"
- Misses variations: `'DROP'+'TABLE'`, encoded versions, etc.

**With ML:**
- Learns from thousands of examples
- Detects subtle patterns
- Adapts to new attack types
- Reduces false positives

## 📈 What We Learn

By analyzing HTTP requests, the system learns:
- Normal requests have predictable patterns
- Attacks have statistical anomalies
- Certain combinations of features are highly suspicious
- The model becomes better at detecting new threats

---

**Next**: [Project Overview](project-overview.md) - How all these pieces fit together!


