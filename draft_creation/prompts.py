import logging

prompts = {
    "specs_system": """
    You are an AI assistant tasked with generating specifications for code changes. Based on the original code provided and the git-style diff, please perform the following step:
    **Create an Specs Snippet**
   - Consider the original code and the diff input provided and generate the most likely code change specification that the diff implements.
   - The original code file will be enclosed within `<original_code>` tags.
   - The diff will be enclosed within `<diff>` tags.
   - Enclose the specifictions within `<specs>` tags.
   
   **Instructions**
    - Do not include any explanations or commentary outside of the specified tag.
    - Make sure that the specifications are clear and concise.
    - Make sure that the specifications {detail_level}.

    **Example**
    {example}
    """,

    "specs_user":
    """
    <original_code>
    {original_code}
    </original_code>

    <diff>
    {diff}
    </diff>
    
    """,

    "draft_from_specs_system": """
    You are an AI assistant tasked with generating synthetic code updates for model training purposes. Based on the original code and specifications provided, please perform the following step:
    **Create an Update Snippet**
   - Modify the original code as specified (e.g add features, remove code).
   - Include only the new or changed code.
   - Use the exact ellipsis comment `// ... existing code ...` to represent omitted unchanged lines.
   - Focus only on the relevant parts; do not include the entire code.
   - Ensure the update snippet is concise and clearly shows where changes are applied.
   - Enclose the update snippet within `<update_snippet>` tags.
   
   **Instructions**
   - Do not include any explanations or commentary outside of the specified tags.
   
    **Example**
    {example}
 """,
    "draft_from_specs_user":
    """
    <original_code>
    {original_code}
    </original_code>

    <specs>
    {specs}
    </specs>
    """,

    "draft_from_diff_system": """
    You are an AI assistant tasked with generating synthetic code updates for model training purposes. Based on the original code and git-style diff provided, please perform the following step:
    **Create an Update Snippet**
   - Modify the original code as shown in the git-style diff.\
   - Include only the new or changed code.
   - Use the exact ellipsis comment `// ... existing code ...` to represent omitted unchanged lines.
   - Focus only on the relevant parts; do not include the entire code.
   - Ensure the update snippet is concise and clearly shows where changes are applied.
   - Enclose the update snippet within `<update_snippet>` tags.
   - You are generating a draft of a code snippet and not a diff, so do not use symbols like `+` or `-` to indicate additions or deletions.
   
   **Instructions**
    - Do not include any explanations or commentary outside of the specified tags.

    **Example**
    {example}
    """,

    "draft_from_diff_user":
    """
    
    **Original Code**
    <original_code>
    {original_code}
    </original_code>
    
    **Diff**
    <diff>
    {diff}
    </diff>
    """
}

in_context_example = {
    "detailed_spec": """
    <original_code>
    def process_grades(filepath):
    def calculate_grade(score):
        if score >= 90: return 'A'
        if score >= 80: return 'B'
        if score >= 70: return 'C'
        if score >= 60: return 'D'
        return 'F'

    with open(filepath, 'r') as file:
        students = {}
        for line in file:
            name, score = line.strip().split(',')
            score = int(score)
            grade = calculate_grade(score)
            students[name] = {'score': score, 'grade': grade}
        
        return students

    if __name__ == "__main__":
        results = process_grades("grades.txt")
        for student, data in results.items():
            print(f"Student: {student}, Score: {data['score']}, Grade: {data['grade']}")
    </original_code>
    
    <diff>
    diff --git a/grade_processor.py b/grade_processor.py
index abcd123..efgh456 789
--- a/grade_processor.py 
+++ b/grade_processor.py
@@ -1,4 +1,4 @@
-def process_grades(filepath):
+def process_grades(filepath, with_stats=False):
    def calculate_grade(score):
        if score >= 90: return 'A'
        if score >= 80: return 'B'
@@ -13,11 +13,30 @@ def process_grades(filepath):
            grade = calculate_grade(score)
            students[name] = {'score': score, 'grade': grade}
        
+        if with_stats:
+            # Calculate class statistics
+            scores = [s['score'] for s in students.values()]
+            stats = {
+                'average': sum(scores) / len(scores),
+                'highest': max(scores),
+                'lowest': min(scores),
+                'total_students': len(scores)
+            }
+            
+            # Add rankings
+            sorted_students = sorted(students.items(), 
+                                  key=lambda x: x[1]['score'], 
+                                  reverse=True)
+            for rank, (name, data) in enumerate(sorted_students, 1):
+                students[name]['rank'] = rank
+            
+            return students, stats
+        
        return students

    if __name__ == "__main__":
-       results = process_grades("grades.txt")
+       results, stats = process_grades("grades.txt", with_stats=True)
+       print("\nClass Statistics:")
+       for stat, value in stats.items():
+        print(f"{stat}: {value}")
+    
+       print("\nStudent Results:")
        for student, data in results.items():
-           print(f"Student: {student}, Score: {data['score']}, Grade: {data['grade']}")
+           print(f"Student: {student}, Score: {data['score']}, "
+               f"Grade: {data['grade']}, Rank: {data['rank']}")
    </diff>
    <specs>
    # Grade Processing Statistics Feature

## 1. Function Modification
Enhance process_grades() to support statistical analysis

### 1.1 Input Parameters
- filepath: string (existing)
  * Path to grades file
  * No change to format requirements
- with_stats: boolean (new)
  * Optional parameter
  * Default: False
  * Controls statistics generation

### 1.2 Return Values
Basic Mode (with_stats=False):
- Return existing students dictionary
- No changes to current format

Statistics Mode (with_stats=True):
- Return tuple: (students_dict, stats_dict)
- students_dict additions:
  * New 'rank' key for each student
  * Ranks are 1-based integers
- stats_dict structure:
  * average: float (class mean score)
  * highest: int (highest score)
  * lowest: int (lowest score)
  * total_students: int

### 1.3 Data Processing Rules
- Ranking:
  * Based on numerical score
  * Highest score = rank 1
  * Ties receive same rank
  * No gaps in rankings

- Statistics Calculation:
  * Average: mean of all scores
  * No rounding specified
  * Include all students in calculations

### 1.4 File Format Requirements
- No changes to existing format
- Maintains comma-separated format
- First field: student name
- Second field: numerical score

### 1.5 Validation Requirements
- Maintain existing validation
- No additional error cases
- Handle empty file case in statistics

## 2. Output Format
Statistics Mode Output:
- Class statistics printed first
- One statistic per line
- Student results follow
- Include rank in student output

## 3. Constraints
- Preserve all existing functionality
- No changes to grade calculation
- Maintain file reading logic
    </specs>
    """,

    "higher_level_spec":
    """
        <original_code>
    def process_grades(filepath):
    def calculate_grade(score):
        if score >= 90: return 'A'
        if score >= 80: return 'B'
        if score >= 70: return 'C'
        if score >= 60: return 'D'
        return 'F'

    with open(filepath, 'r') as file:
        students = {}
        for line in file:
            name, score = line.strip().split(',')
            score = int(score)
            grade = calculate_grade(score)
            students[name] = {'score': score, 'grade': grade}
        
        return students

    if __name__ == "__main__":
        results = process_grades("grades.txt")
        for student, data in results.items():
            print(f"Student: {student}, Score: {data['score']}, Grade: {data['grade']}")
    </original_code>
    
    <diff>
    diff --git a/grade_processor.py b/grade_processor.py
index abcd123..efgh456 789
--- a/grade_processor.py 
+++ b/grade_processor.py
@@ -1,4 +1,4 @@
-def process_grades(filepath):
+def process_grades(filepath, with_stats=False):
    def calculate_grade(score):
        if score >= 90: return 'A'
        if score >= 80: return 'B'
@@ -13,11 +13,30 @@ def process_grades(filepath):
            grade = calculate_grade(score)
            students[name] = {'score': score, 'grade': grade}
        
+        if with_stats:
+            # Calculate class statistics
+            scores = [s['score'] for s in students.values()]
+            stats = {
+                'average': sum(scores) / len(scores),
+                'highest': max(scores),
+                'lowest': min(scores),
+                'total_students': len(scores)
+            }
+            
+            # Add rankings
+            sorted_students = sorted(students.items(), 
+                                  key=lambda x: x[1]['score'], 
+                                  reverse=True)
+            for rank, (name, data) in enumerate(sorted_students, 1):
+                students[name]['rank'] = rank
+            
+            return students, stats
+        
        return students

    if __name__ == "__main__":
-       results = process_grades("grades.txt")
+       results, stats = process_grades("grades.txt", with_stats=True)
+       print("\nClass Statistics:")
+       for stat, value in stats.items():
+        print(f"{stat}: {value}")
+    
+       print("\nStudent Results:")
        for student, data in results.items():
-           print(f"Student: {student}, Score: {data['score']}, Grade: {data['grade']}")
+           print(f"Student: {student}, Score: {data['score']}, "
+               f"Grade: {data['grade']}, Rank: {data['rank']}")
    </diff>
    <specs>
# Grade Processing Enhancement

## Purpose
Add class statistics and student ranking to existing grade processor

## New Features
- Calculate class average, highest, and lowest scores
- Add student rankings based on scores
- Maintain backwards compatibility
- Optional statistics mode

## Expected Output Changes
- Standard mode: Original grade report
- Statistics mode: Adds class stats and rankings

## Impact
- No changes to existing grade calculation
- New optional statistics without breaking existing implementations
    </specs>
    """,
    "draft_from_specs":"""
    <original_code>
    def process_grades(filepath):
    def calculate_grade(score):
        if score >= 90: return 'A'
        if score >= 80: return 'B'
        if score >= 70: return 'C'
        if score >= 60: return 'D'
        return 'F'

    with open(filepath, 'r') as file:
        students = {}
        for line in file:
            name, score = line.strip().split(',')
            score = int(score)
            grade = calculate_grade(score)
            students[name] = {'score': score, 'grade': grade}
        
        return students

    if __name__ == "__main__":
        results = process_grades("grades.txt")
        for student, data in results.items():
            print(f"Student: {student}, Score: {data['score']}, Grade: {data['grade']}")
    </original_code>
        <specs>
    # Grade Processing Statistics Feature

## 1. Function Modification
Enhance process_grades() to support statistical analysis

### 1.1 Input Parameters
- filepath: string (existing)
  * Path to grades file
  * No change to format requirements
- with_stats: boolean (new)
  * Optional parameter
  * Default: False
  * Controls statistics generation

### 1.2 Return Values
Basic Mode (with_stats=False):
- Return existing students dictionary
- No changes to current format

Statistics Mode (with_stats=True):
- Return tuple: (students_dict, stats_dict)
- students_dict additions:
  * New 'rank' key for each student
  * Ranks are 1-based integers
- stats_dict structure:
  * average: float (class mean score)
  * highest: int (highest score)
  * lowest: int (lowest score)
  * total_students: int

### 1.3 Data Processing Rules
- Ranking:
  * Based on numerical score
  * Highest score = rank 1
  * Ties receive same rank
  * No gaps in rankings

- Statistics Calculation:
  * Average: mean of all scores
  * No rounding specified
  * Include all students in calculations

### 1.4 File Format Requirements
- No changes to existing format
- Maintains comma-separated format
- First field: student name
- Second field: numerical score

### 1.5 Validation Requirements
- Maintain existing validation
- No additional error cases
- Handle empty file case in statistics

## 2. Output Format
Statistics Mode Output:
- Class statistics printed first
- One statistic per line
- Student results follow
- Include rank in student output

## 3. Constraints
- Preserve all existing functionality
- No changes to grade calculation
- Maintain file reading logic
    </specs>
<update_snippet>
def process_grades(filepath, with_stats=False):
    def calculate_grade(score):
    // ... existing code ...

        if with_stats:
            # Calculate class statistics
            scores = [s['score'] for s in students.values()]
            stats = {
                'average': sum(scores) / len(scores),
                'highest': max(scores),
                'lowest': min(scores),
                'total_students': len(scores)
            }
            
            # Add rankings
            sorted_students = sorted(students.items(), 
                                  key=lambda x: x[1]['score'], 
                                  reverse=True)
            for rank, (name, data) in enumerate(sorted_students, 1):
                students[name]['rank'] = rank
            
            return students, stats
        
        return students

if __name__ == "__main__":
    results, stats = process_grades("grades.txt", with_stats=True)
    print("\nClass Statistics:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    
    print("\nStudent Results:")
    for student, data in results.items():
        print(f"Student: {student}, Score: {data['score']}, "
              f"Grade: {data['grade']}, Rank: {data['rank']}")
</update_snippet>
    """,
    "draft_from_diff":"""
    <original_code>
    def process_grades(filepath):
    def calculate_grade(score):
        if score >= 90: return 'A'
        if score >= 80: return 'B'
        if score >= 70: return 'C'
        if score >= 60: return 'D'
        return 'F'

    with open(filepath, 'r') as file:
        students = {}
        for line in file:
            name, score = line.strip().split(',')
            score = int(score)
            grade = calculate_grade(score)
            students[name] = {'score': score, 'grade': grade}
        
        return students

    if __name__ == "__main__":
        results = process_grades("grades.txt")
        for student, data in results.items():
            print(f"Student: {student}, Score: {data['score']}, Grade: {data['grade']}")
    </original_code>
    <diff>
    diff --git a/grade_processor.py b/grade_processor.py
index abcd123..efgh456 789
--- a/grade_processor.py 
+++ b/grade_processor.py
@@ -1,4 +1,4 @@
-def process_grades(filepath):
+def process_grades(filepath, with_stats=False):
    def calculate_grade(score):
        if score >= 90: return 'A'
        if score >= 80: return 'B'
@@ -13,11 +13,30 @@ def process_grades(filepath):
            grade = calculate_grade(score)
            students[name] = {'score': score, 'grade': grade}
        
+        if with_stats:
+            # Calculate class statistics
+            scores = [s['score'] for s in students.values()]
+            stats = {
+                'average': sum(scores) / len(scores),
+                'highest': max(scores),
+                'lowest': min(scores),
+                'total_students': len(scores)
+            }
+            
+            # Add rankings
+            sorted_students = sorted(students.items(), 
+                                  key=lambda x: x[1]['score'], 
+                                  reverse=True)
+            for rank, (name, data) in enumerate(sorted_students, 1):
+                students[name]['rank'] = rank
+            
+            return students, stats
+        
        return students

    if __name__ == "__main__":
-       results = process_grades("grades.txt")
+       results, stats = process_grades("grades.txt", with_stats=True)
+       print("\nClass Statistics:")
+       for stat, value in stats.items():
+        print(f"{stat}: {value}")
+    
+       print("\nStudent Results:")
        for student, data in results.items():
-           print(f"Student: {student}, Score: {data['score']}, Grade: {data['grade']}")
+           print(f"Student: {student}, Score: {data['score']}, "
+               f"Grade: {data['grade']}, Rank: {data['rank']}")
    </diff>
<update_snippet>
def process_grades(filepath, with_stats=False):
    def calculate_grade(score):
    // ... existing code ...

        if with_stats:
            # Calculate class statistics
            scores = [s['score'] for s in students.values()]
            stats = {
                'average': sum(scores) / len(scores),
                'highest': max(scores),
                'lowest': min(scores),
                'total_students': len(scores)
            }
            
            # Add rankings
            sorted_students = sorted(students.items(), 
                                  key=lambda x: x[1]['score'], 
                                  reverse=True)
            for rank, (name, data) in enumerate(sorted_students, 1):
                students[name]['rank'] = rank
            
            return students, stats
        
        return students

if __name__ == "__main__":
    results, stats = process_grades("grades.txt", with_stats=True)
    print("\nClass Statistics:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    
    print("\nStudent Results:")
    for student, data in results.items():
        print(f"Student: {student}, Score: {data['score']}, "
              f"Grade: {data['grade']}, Rank: {data['rank']}")
</update_snippet>
    """
}

def get_prompt_chat(prompt_name,original_code="",diff="",specs=""):
    if prompt_name == "detailed_specs":
        return [
            {'role': 'system', 'content': prompts["specs_system"]
            .format(detail_level="are detailed and verbose",
            example=in_context_example["detailed_spec"])},
            {'role': 'user', 'content': prompts["specs_user"]
            .format(original_code=original_code,
            diff=diff)}
        ]
    elif prompt_name == "higher_level_specs":
        return [
            {'role': 'system', 'content': prompts["specs_system"]
            .format(detail_level="are high-level and concise but but do not miss out on details that are important for the code implementation",
            example=in_context_example["higher_level_spec"])},
            {'role': 'user', 'content': prompts["specs_user"]
            .format(original_code=original_code,
            diff=diff)}
        ]
    elif prompt_name == "draft_from_specs":
        return [
            {'role': 'system', 'content': prompts["draft_from_specs_system"]
            .format(example=in_context_example["draft_from_specs"])},
            {'role': 'user', 'content': prompts["draft_from_specs_user"]
            .format(original_code=original_code,
                    specs=specs)}
        ]
    elif prompt_name == "draft_from_diff":
        return [
            {'role': 'system', 'content': prompts["draft_from_diff_system"]
            .format(example=in_context_example["draft_from_diff"])},
            {'role': 'user', 'content': prompts["draft_from_diff_user"]
            .format(original_code=original_code,
                    diff=diff)}
        ]
    else:
        logging.warning(f"Prompt name '{prompt_name}' not recognized.")
        return None