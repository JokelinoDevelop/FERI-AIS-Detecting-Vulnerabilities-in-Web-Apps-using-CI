"""
Feature engineering for HTTP request data
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs
import logging

logger = logging.getLogger(__name__)


class HTTPFeatureEngineer:
    """Feature engineering for HTTP request classification"""

    def __init__(self):
        """Initialize feature engineer"""
        # Suspicious patterns commonly found in web attacks
        self.suspicious_patterns = {
            'sql_injection': [
                r'union\s+select', r'1=1', r'--', r'/\*', r'\*/',
                r'xp_cmdshell', r'exec', r'cast\s*\('
            ],
            'xss': [
                r'<script', r'javascript:', r'onload=', r'onerror=',
                r'<iframe', r'<object', r'<embed'
            ],
            'path_traversal': [
                r'\.\./', r'\.\.\\', r'%2e%2e%2f', r'%2e%2e%5c'
            ],
            'command_injection': [
                r';\s*', r'`', r'\$\(', r'\|', r'&'
            ]
        }

        # Common HTTP methods
        self.http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH']

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features from HTTP request data for detailed vulnerability analysis

        Args:
            df: DataFrame with HTTP request data

        Returns:
            DataFrame with extracted features
        """
        logger.info("Extracting comprehensive features from HTTP requests...")

        # Create feature DataFrame
        features = pd.DataFrame(index=df.index)

        # Method features (full one-hot encoding with all standard methods)
        features = pd.concat([features, self._extract_method_features(df)], axis=1)

        # URL features (comprehensive URL analysis)
        features = pd.concat([features, self._extract_url_features(df)], axis=1)

        # User-Agent features (browser and attack tool detection)
        features = pd.concat([features, self._extract_user_agent_features(df)], axis=1)

        # Content features (request body analysis)
        features = pd.concat([features, self._extract_content_features(df)], axis=1)

        # Header features (HTTP header analysis)
        features = pd.concat([features, self._extract_header_features(df)], axis=1)

        # Suspicious pattern features (comprehensive attack pattern detection)
        features = pd.concat([features, self._extract_suspicious_features(df)], axis=1)

        logger.info(f"Extracted {features.shape[1]} comprehensive features")
        return features

    def _extract_method_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from HTTP method"""
        features = pd.DataFrame(index=df.index)

        # One-hot encode HTTP methods with fixed categories to ensure consistency
        method_dummies = pd.get_dummies(df['Method'], prefix='method')
        # Ensure all expected methods are present (even if with all zeros)
        for method in self.http_methods:
            col_name = f'method_{method}'
            if col_name not in method_dummies.columns:
                method_dummies[col_name] = 0

        # Reorder columns to ensure consistency
        method_cols = [f'method_{method}' for method in self.http_methods]
        method_dummies = method_dummies[method_cols]

        features = pd.concat([features, method_dummies], axis=1)

        # Method is standard (1) or not (0)
        features['method_is_standard'] = df['Method'].isin(self.http_methods).astype(int)

        return features

    def _extract_method_features_simplified(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract simplified method features (only most common methods)"""
        features = pd.DataFrame(index=df.index)

        # Only keep GET, POST, PUT (DELETE and others have very low importance)
        common_methods = ['GET', 'POST', 'PUT']
        for method in common_methods:
            features[f'method_{method}'] = (df['Method'] == method).astype(int)

        return features

    def _extract_suspicious_features_simplified(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract simplified suspicious pattern features (only most important ones)"""
        features = pd.DataFrame(index=df.index)

        # Only SQL injection patterns (most important) and XSS patterns (moderately important)
        # Skip path traversal and command injection as they have very low importance

        # Check for SQL injection patterns
        sql_patterns = '|'.join(self.suspicious_patterns['sql_injection'])
        features['has_sql_injection_patterns'] = df['URL'].str.contains(
            sql_patterns, case=False, na=False
        ).astype(int) | df['content'].str.contains(
            sql_patterns, case=False, na=False
        ).astype(int)

        # Check for XSS patterns
        xss_patterns = '|'.join(self.suspicious_patterns['xss'])
        features['has_xss_patterns'] = df['URL'].str.contains(
            xss_patterns, case=False, na=False
        ).astype(int) | df['content'].str.contains(
            xss_patterns, case=False, na=False
        ).astype(int)

        return features

    def _extract_url_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from URL"""
        features = pd.DataFrame(index=df.index)

        # URL length
        features['url_length'] = df['URL'].str.len()

        # URL path length (after domain)
        def get_path_length(url):
            try:
                parsed = urlparse(str(url))
                return len(parsed.path)
            except:
                return 0

        features['url_path_length'] = df['URL'].apply(get_path_length)

        # Query parameter count
        def get_query_param_count(url):
            try:
                parsed = urlparse(str(url))
                params = parse_qs(parsed.query)
                return len(params)
            except:
                return 0

        features['url_query_param_count'] = df['URL'].apply(get_query_param_count)

        # Query parameter total length
        def get_query_length(url):
            try:
                parsed = urlparse(str(url))
                return len(parsed.query)
            except:
                return 0

        features['url_query_length'] = df['URL'].apply(get_query_length)

        # Contains suspicious characters in URL
        suspicious_url_chars = ['<', '>', '"', "'", ';', '|', '&', '$', '`']
        features['url_has_suspicious_chars'] = df['URL'].apply(
            lambda x: any(char in str(x) for char in suspicious_url_chars)
        ).astype(int)

        # URL contains encoded characters
        features['url_has_encoded_chars'] = df['URL'].str.contains(r'%[0-9A-Fa-f]{2}').astype(int)

        return features

    def _extract_user_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from User-Agent"""
        features = pd.DataFrame(index=df.index)

        # User-Agent length
        features['user_agent_length'] = df['User-Agent'].str.len()

        # User-Agent is empty
        features['user_agent_empty'] = df['User-Agent'].str.strip().eq('').astype(int)

        # Common browser patterns
        common_browsers = ['Mozilla', 'Chrome', 'Safari', 'Firefox', 'Edge', 'Opera']
        features['user_agent_common_browser'] = df['User-Agent'].apply(
            lambda x: any(browser in str(x) for browser in common_browsers)
        ).astype(int)

        # Contains suspicious patterns
        suspicious_ua_patterns = ['sqlmap', 'nmap', 'nikto', 'dirbuster', 'gobuster']
        features['user_agent_suspicious'] = df['User-Agent'].apply(
            lambda x: any(pattern.lower() in str(x).lower() for pattern in suspicious_ua_patterns)
        ).astype(int)

        return features

    def _extract_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from request content"""
        features = pd.DataFrame(index=df.index)

        # Content length
        features['content_length'] = df['content_length']

        # Content is empty
        features['content_empty'] = (df['content_length'] == 0).astype(int)

        # Content contains suspicious characters
        suspicious_content_chars = ['<', '>', '"', "'", ';', '|', '&', '$', '`']
        features['content_has_suspicious_chars'] = df['content'].apply(
            lambda x: any(char in str(x) for char in suspicious_content_chars)
        ).astype(int)

        # Content has encoded characters
        features['content_has_encoded_chars'] = df['content'].str.contains(r'%[0-9A-Fa-f]{2}').astype(int)

        return features

    def _extract_header_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from HTTP headers"""
        features = pd.DataFrame(index=df.index)

        # Check presence of important headers
        important_headers = ['Pragma', 'Cache-Control', 'Accept', 'Accept-encoding',
                           'Accept-charset', 'language', 'host', 'cookie', 'content-type']

        for header in important_headers:
            if header in df.columns:
                features[f'header_{header.lower()}_present'] = (~df[header].isna() &
                                                              (df[header].str.strip() != '')).astype(int)

        # Host header analysis
        features['host_localhost'] = df['host'].str.contains('localhost').astype(int)

        # Language header analysis
        features['language_english'] = df['language'].str.contains('en').astype(int)

        return features

    def _extract_suspicious_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features based on suspicious patterns"""
        features = pd.DataFrame(index=df.index)

        # Check for SQL injection patterns
        sql_patterns = '|'.join(self.suspicious_patterns['sql_injection'])
        features['has_sql_injection_patterns'] = df['URL'].str.contains(
            sql_patterns, case=False, na=False
        ).astype(int) | df['content'].str.contains(
            sql_patterns, case=False, na=False
        ).astype(int)

        # Check for XSS patterns
        xss_patterns = '|'.join(self.suspicious_patterns['xss'])
        features['has_xss_patterns'] = df['URL'].str.contains(
            xss_patterns, case=False, na=False
        ).astype(int) | df['content'].str.contains(
            xss_patterns, case=False, na=False
        ).astype(int)

        # Check for path traversal patterns
        traversal_patterns = '|'.join(self.suspicious_patterns['path_traversal'])
        features['has_path_traversal_patterns'] = df['URL'].str.contains(
            traversal_patterns, case=False, na=False
        ).astype(int) | df['content'].str.contains(
            traversal_patterns, case=False, na=False
        ).astype(int)

        # Check for command injection patterns
        cmd_patterns = '|'.join(self.suspicious_patterns['command_injection'])
        features['has_command_injection_patterns'] = df['URL'].str.contains(
            cmd_patterns, case=False, na=False
        ).astype(int) | df['content'].str.contains(
            cmd_patterns, case=False, na=False
        ).astype(int)

        # Overall suspicious score (count of different attack types detected)
        suspicious_cols = [col for col in features.columns if col.startswith('has_')]
        features['suspicious_pattern_score'] = features[suspicious_cols].sum(axis=1)

        return features
