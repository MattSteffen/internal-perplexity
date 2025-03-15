"""
CSV Document Reader

This module provides the CSVReader class for reading and extracting content from
CSV files.
"""

import os
import csv
from typing import Any, Optional, List

from .base import DocumentReader
from ..document_content import DocumentContent


class CSVReader(DocumentReader):
    """Reader for CSV files."""
    
    def read(self, file_path: str) -> DocumentContent:
        """Read CSV file and extract content.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DocumentContent object containing the extracted content
        """
        content = DocumentContent()
        try:
            # Try to detect the dialect first
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                sample = f.read(1024)
                sniffer = csv.Sniffer()
                has_header = sniffer.has_header(sample)
                dialect = sniffer.sniff(sample)
            
            # Now read the CSV with the detected dialect
            with open(file_path, 'r', encoding='utf-8', errors='replace', newline='') as f:
                reader = csv.reader(f, dialect=dialect)
                rows = list(reader)
                
                # Extract header row if detected
                header_row = None
                data_rows = rows
                if has_header and rows:
                    header_row = rows[0]
                    data_rows = rows[1:]
                
                # Calculate basic statistics
                row_count = len(data_rows)
                column_count = len(header_row) if header_row else (len(rows[0]) if rows else 0)
                
                # Format the CSV data as text
                formatted_rows = []
                if header_row:
                    formatted_rows.append(" | ".join(header_row))
                    formatted_rows.append("-" * (sum(len(h) + 3 for h in header_row)))
                
                for row in data_rows:
                    formatted_rows.append(" | ".join(row))
                
                formatted_csv = "\n".join(formatted_rows)
                content.add_text(formatted_csv)
                
                # Add as table as well
                content.add_table(formatted_csv)
                
                # Set metadata
                content.set_metadata({
                    'source': file_path,
                    'format': 'csv',
                    'size_bytes': os.path.getsize(file_path),
                    'rows': row_count + (1 if has_header else 0),
                    'columns': column_count,
                    'has_header': has_header
                })
                
                # Try to determine data types in each column
                if rows:
                    column_types = self._analyze_column_types(data_rows, column_count)
                    content.update_metadata({'column_types': column_types})
                
            return content
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            content.add_text(f"Error reading file: {str(e)}")
            return content
    
    def _analyze_column_types(self, data_rows: List[List[str]], column_count: int) -> List[str]:
        """Analyze and determine likely data types for each column.
        
        Args:
            data_rows: List of data rows
            column_count: Number of columns
            
        Returns:
            List of column type strings
        """
        column_types = []
        
        for col_idx in range(column_count):
            # Get all values in this column
            column_values = [row[col_idx] if col_idx < len(row) else "" for row in data_rows]
            
            # Check if all values can be converted to numbers
            numeric_count = 0
            date_count = 0
            empty_count = 0
            
            for value in column_values:
                if not value.strip():
                    empty_count += 1
                    continue
                    
                # Try to convert to float
                try:
                    float(value)
                    numeric_count += 1
                    continue
                except ValueError:
                    pass
                
                # Try basic date format detection
                # This is a simplified approach - in a real app, would use more sophisticated date detection
                if len(value) >= 8 and (('/' in value) or ('-' in value) or ('.' in value)):
                    date_count += 1
            
            # Determine column type based on majority
            non_empty_count = len(column_values) - empty_count
            if non_empty_count == 0:
                column_type = "empty"
            elif numeric_count / non_empty_count > 0.8:
                column_type = "numeric"
            elif date_count / non_empty_count > 0.8:
                column_type = "date"
            else:
                column_type = "text"
                
            column_types.append(column_type)
            
        return column_types 