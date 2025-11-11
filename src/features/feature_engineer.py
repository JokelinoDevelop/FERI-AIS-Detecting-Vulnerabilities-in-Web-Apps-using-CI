"""
Feature engineering for HTTP request data
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qs, unquote
import logging

from ..utils.config import get_logger, FeatureEngineeringError

logger = get_logger(__name__)


class HTTPFeatureEngineer:
    """Feature engineering for HTTP request classification"""

    def __init__(self):
        """Initialize feature engineer"""
        # Suspicious patterns commonly found in web attacks
        self.suspicious_patterns = {
            'sql_injection': [
                r'union\s+select', r'1=1', r'--', r'/\*', r'\*/',
                r'xp_cmdshell', r'exec', r'cast\s*\(', r'information_schema',
                r'concat\s*\(', r'group_concat', r'select\s+.*\s+from'
            ],
            'xss': [
                r'<script', r'javascript:', r'onload=', r'onerror=',
                r'<iframe', r'<object', r'<embed', r'document\.cookie',
                r'alert\s*\(', r'eval\s*\(', r'<img.*onerror'
            ],
            'path_traversal': [
                r'\.\./', r'\.\.\\', r'%2e%2e%2f', r'%2e%2e%5c',
                r'\.\.%2f', r'\.\.%5c', r'etc/passwd', r'boot\.ini'
            ],
            'command_injection': [
                r';\s*', r'`', r'\$\(', r'\$\{', r'\|', r'&',
                r'wget\s+', r'curl\s+', r'ping\s+-c'
            ]
        }

        # Common HTTP methods
        self.http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH']

        # Compile regex patterns for efficiency
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for better performance"""
        compiled = {}
        for attack_type, patterns in self.suspicious_patterns.items():
            compiled[attack_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return compiled

    def _safe_search(self, text: str, patterns: List[re.Pattern]) -> bool:
        """Safely search for patterns in text"""
        if not isinstance(text, str) or not text.strip():
            return False
        return any(pattern.search(text) for pattern in patterns)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from HTTP request data

        Args:
            df: DataFrame with HTTP request data

        Returns:
            DataFrame with extracted features

        Raises:
            ValueError: If required columns are missing from DataFrame
        """
        logger.info("Extracting features from HTTP requests...")

        # Validate required columns
        required_columns = ['Method', 'URL', 'User-Agent', 'content', 'content_length']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        try:
            # Create feature DataFrame
            features = pd.DataFrame(index=df.index)

            # Extract features from different sources
            feature_extractors = [
                self._extract_method_features,
                self._extract_url_features,
                self._extract_user_agent_features,
                self._extract_content_features,
                self._extract_header_features,
                self._extract_suspicious_features
            ]

            for extractor in feature_extractors:
                try:
                    new_features = extractor(df)
                    features = pd.concat([features, new_features], axis=1)
                except Exception as e:
                    logger.warning(f"Failed to extract features with {extractor.__name__}: {e}")
                    continue

            logger.info(f"Extracted {features.shape[1]} features from {len(df)} samples")
            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise FeatureEngineeringError(f"Feature extraction failed: {e}") from e

    def _extract_method_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from HTTP method"""
        features = pd.DataFrame(index=df.index)

        # One-hot encode HTTP methods
        method_dummies = pd.get_dummies(df['Method'], prefix='method')
        features = pd.concat([features, method_dummies], axis=1)

        # Method is standard (1) or not (0)
        features['method_is_standard'] = df['Method'].isin(self.http_methods).astype(int)

        return features

    def _extract_url_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from URL"""
        features = pd.DataFrame(index=df.index)

        # URL length (handle None/NaN values)
        features['url_length'] = df['URL'].fillna('').str.len()

        # URL path length (after domain)
        def get_path_length(url: str) -> int:
            """Safely extract path length from URL"""
            try:
                if not isinstance(url, str) or not url.strip():
                    return 0
                parsed = urlparse(url)
                return len(parsed.path)
            except Exception:
                return 0

        features['url_path_length'] = df['URL'].apply(get_path_length)

        # Query parameter count
        def get_query_param_count(url: str) -> int:
            """Safely count query parameters"""
            try:
                if not isinstance(url, str) or not url.strip():
                    return 0
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                return len(params)
            except Exception:
                return 0

        features['url_query_param_count'] = df['URL'].apply(get_query_param_count)

        # Query parameter total length
        def get_query_length(url: str) -> int:
            """Safely get query string length"""
            try:
                if not isinstance(url, str) or not url.strip():
                    return 0
                parsed = urlparse(url)
                return len(parsed.query)
            except Exception:
                return 0

        features['url_query_length'] = df['URL'].apply(get_query_length)

        # URL structure features
        def get_url_depth(url: str) -> int:
            """Get URL path depth"""
            try:
                if not isinstance(url, str) or not url.strip():
                    return 0
                parsed = urlparse(url)
                path_parts = [p for p in parsed.path.split('/') if p]
                return len(path_parts)
            except Exception:
                return 0

        features['url_depth'] = df['URL'].apply(get_url_depth)

        # Contains suspicious characters in URL
        suspicious_url_chars = ['<', '>', '"', "'", ';', '|', '$', '`', '\\', '\x00']
        features['url_has_suspicious_chars'] = df['URL'].fillna('').apply(
            lambda x: any(char in x for char in suspicious_url_chars)
        ).astype(int)

        # URL contains encoded characters (%XX)
        features['url_has_encoded_chars'] = df['URL'].fillna('').str.contains(
            r'%[0-9A-Fa-f]{2}', regex=True
        ).astype(int)

        # URL contains double encoding
        features['url_has_double_encoding'] = df['URL'].fillna('').str.contains(
            r'%25[0-9A-Fa-f]{2}', regex=True
        ).astype(int)

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

        # Check for different attack patterns using compiled regex
        attack_types = ['sql_injection', 'xss', 'path_traversal', 'command_injection']

        for attack_type in attack_types:
            feature_name = f'has_{attack_type}_patterns'

            # Check URL and content for patterns
            url_matches = df['URL'].fillna('').apply(
                lambda x: self._safe_search(x, self._compiled_patterns[attack_type])
            ).astype(int)

            content_matches = df['content'].fillna('').apply(
                lambda x: self._safe_search(x, self._compiled_patterns[attack_type])
            ).astype(int)

            # Combine URL and content matches
            features[feature_name] = (url_matches | content_matches).astype(int)

        # Additional suspicious pattern features
        features['has_mixed_encoding'] = df['URL'].fillna('').apply(
            lambda x: bool(re.search(r'%[0-9A-Fa-f]{2}.*%[0-9A-Fa-f]{2}', x, re.IGNORECASE))
        ).astype(int)

        # Check for common attack signatures in User-Agent
        ua_attack_patterns = [r'sqlmap', r'nmap', r'nikto', r'gobuster', r'metasploit', r'burpsuite']
        ua_compiled = [re.compile(pattern, re.IGNORECASE) for pattern in ua_attack_patterns]
        features['user_agent_attack_tool'] = df['User-Agent'].fillna('').apply(
            lambda x: self._safe_search(x, ua_compiled)
        ).astype(int)

        # Overall suspicious score (count of different attack types detected)
        suspicious_cols = [col for col in features.columns if col.startswith('has_')]
        features['suspicious_pattern_score'] = features[suspicious_cols].sum(axis=1)

        # High risk score (multiple attack types detected)
        features['high_risk_score'] = (features['suspicious_pattern_score'] >= 2).astype(int)

        return features
