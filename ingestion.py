"""
ECB Data Portal SDMX XML Ingestion to PostgreSQL
Implements proper schema with fact tables: FSI, MMR, YC, INF, IR, LFI, MPD
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from lxml import etree
import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    DateTime,
    Integer,
    Text,
    MetaData,
    Table,
    inspect,
    text,
    ForeignKey,
    Date,
)
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SDMX 2.1 XML namespaces
NAMESPACES = {
    'message': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message',
    'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic',
    'common': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common'
}

# Dataset to table mapping
DATASET_TABLE_MAP = {
    'ECB_FMD2': {
        'CISS': 'FSI',  # Financial Stress Indicators
        'YC': 'YC',      # Yield Curves
        'FM': 'MMR',     # Money Market Rates
    },
    'ECB_ICP3': 'INF',      # Inflation
    'ECB_MIR1': 'IR',       # Interest Rates (Banking)
    'EUROSTAT_LFS1': 'LFI', # Labor Force
    'ECB_MPD1': 'MPD',      # Monetary Policy Projections
}


class ECBDataIngestion:
    """Handles ingestion of ECB SDMX XML files into PostgreSQL with proper schema."""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the ingestion handler."""
        load_dotenv()

        if connection_string is None:
            connection_string = os.getenv('DATABASE_URL')
            if connection_string is None:
                raise ValueError(
                    "No database connection string provided. "
                    "Set DATABASE_URL environment variable or pass connection_string."
                )

        self.engine = create_engine(connection_string)
        self.metadata = MetaData()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        logger.info("Database connection established")

    def create_schema(self):
        """Create all tables for the ECB data schema."""

        # 1. Dimension Tables
        dim_frequency = Table('dim_frequency', self.metadata,
            Column('freq_code', String(1), primary_key=True),
            Column('freq_name', String(50)),
            Column('description', String(255))
        )

        dim_reference_area = Table('dim_reference_area', self.metadata,
            Column('area_code', String(10), primary_key=True),
            Column('area_name', String(255)),
            Column('area_type', String(50))
        )

        dim_currency = Table('dim_currency', self.metadata,
            Column('currency_code', String(10), primary_key=True),
            Column('currency_name', String(100))
        )

        dim_unit = Table('dim_unit', self.metadata,
            Column('unit_code', String(20), primary_key=True),
            Column('unit_name', String(100)),
            Column('description', String(255))
        )

        dim_adjustment = Table('dim_adjustment', self.metadata,
            Column('adjustment_code', String(1), primary_key=True),
            Column('adjustment_name', String(100))
        )

        dim_data_type = Table('dim_data_type', self.metadata,
            Column('data_type_code', String(20), primary_key=True),
            Column('data_type_name', String(100)),
            Column('category', String(50))
        )

        # 2. Core Metadata Tables
        series_metadata = Table('series_metadata', self.metadata,
            Column('series_id', Integer, primary_key=True, autoincrement=True),
            Column('series_key', String(500), unique=True, nullable=False),
            Column('dataset_code', String(50)),
            Column('fact_table', String(10)),
            Column('title', Text),
            Column('title_complete', Text),
            Column('unit_code', String(20)),
            Column('unit_mult', Integer),
            Column('decimals', Integer),
            Column('frequency_code', String(1)),
            Column('time_format', String(20)),
            Column('first_obs_date', String(50)),
            Column('last_obs_date', String(50)),
            Column('total_obs', Integer),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        )

        fact_table_metadata = Table('fact_table_metadata', self.metadata,
            Column('table_code', String(10), primary_key=True),
            Column('table_name', String(100)),
            Column('description', Text),
            Column('dataset_source', String(50)),
            Column('series_count', Integer),
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        # 3. Fact Tables (use lowercase for PostgreSQL compatibility)
        # FSI - Financial Stress Indicators
        fsi = Table('fsi', self.metadata,
            Column('obs_id', Integer, primary_key=True, autoincrement=True),
            Column('series_id', Integer, ForeignKey('series_metadata.series_id')),
            Column('obs_date', String(50), nullable=False, index=True),
            Column('obs_value', Float),
            Column('obs_status', String(10)),
            Column('obs_conf', String(10)),
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        # MMR - Money Market Rates
        mmr = Table('mmr', self.metadata,
            Column('obs_id', Integer, primary_key=True, autoincrement=True),
            Column('series_id', Integer, ForeignKey('series_metadata.series_id')),
            Column('obs_date', String(50), nullable=False, index=True),
            Column('obs_value', Float),
            Column('obs_status', String(10)),
            Column('obs_conf', String(10)),
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        # YC - Yield Curves
        yc = Table('yc', self.metadata,
            Column('obs_id', Integer, primary_key=True, autoincrement=True),
            Column('series_id', Integer, ForeignKey('series_metadata.series_id')),
            Column('obs_date', String(50), nullable=False, index=True),
            Column('obs_value', Float),
            Column('obs_status', String(10)),
            Column('obs_conf', String(10)),
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        # INF - Inflation Harmonised Indices
        inf = Table('inf', self.metadata,
            Column('obs_id', Integer, primary_key=True, autoincrement=True),
            Column('series_id', Integer, ForeignKey('series_metadata.series_id')),
            Column('obs_date', String(50), nullable=False, index=True),
            Column('obs_value', Float),
            Column('obs_status', String(10)),
            Column('obs_conf', String(10)),
            Column('comment', Text),
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        # IR - Banking Interest Rates
        ir = Table('ir', self.metadata,
            Column('obs_id', Integer, primary_key=True, autoincrement=True),
            Column('series_id', Integer, ForeignKey('series_metadata.series_id')),
            Column('obs_date', String(50), nullable=False, index=True),
            Column('obs_value', Float),
            Column('obs_status', String(10)),
            Column('obs_conf', String(10)),
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        # LFI - Labor Force Indicators
        lfi = Table('lfi', self.metadata,
            Column('obs_id', Integer, primary_key=True, autoincrement=True),
            Column('series_id', Integer, ForeignKey('series_metadata.series_id')),
            Column('obs_date', String(50), nullable=False, index=True),
            Column('obs_value', Float),
            Column('obs_status', String(10)),
            Column('obs_conf', String(10)),
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        # MPD - Monetary Policy Projections
        mpd = Table('mpd', self.metadata,
            Column('obs_id', Integer, primary_key=True, autoincrement=True),
            Column('series_id', Integer, ForeignKey('series_metadata.series_id')),
            Column('obs_date', String(50), nullable=False, index=True),
            Column('obs_value', Float),
            Column('obs_status', String(10)),
            Column('obs_conf', String(10)),
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        # GPR - Geopolitical Risk Indicators
        gpr = Table('gpr', self.metadata,
            Column('obs_id', Integer, primary_key=True, autoincrement=True),
            Column('series_id', Integer, ForeignKey('series_metadata.series_id')),
            Column('obs_date', String(50), nullable=False, index=True),
            Column('obs_value', Float),
            Column('obs_status', String(10)),
            Column('obs_conf', String(10)),
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        # Create all tables
        self.metadata.create_all(self.engine)
        logger.info("Schema created successfully with 8 fact tables + metadata tables")

        # Initialize fact_table_metadata
        self._initialize_fact_table_metadata()
        self._initialize_dimension_data()

    def _initialize_fact_table_metadata(self):
        """Populate fact_table_metadata with descriptions."""
        fact_tables = [
            ('FSI', 'Financial Stress Indicators',
             'CISS - Composite Indicator of Systemic Stress and market segment contributions',
             'ECB_FMD2', 8),
            ('MMR', 'Money Market Rates',
             'Euribor rates for various tenors (1M, 3M, 6M, 1Y)',
             'ECB_FMD2', 4),
            ('YC', 'Yield Curves',
             'AAA-rated government bond spot rates across maturities',
             'ECB_FMD2', 3),
            ('INF', 'Inflation Harmonised Indices',
             'HICP inflation measures (total and energy)',
             'ECB_ICP3', 2),
            ('IR', 'Banking Interest Rates',
             'MFI interest rates for deposits and loans',
             'ECB_MIR1', 4),
            ('LFI', 'Labor Force Indicators',
             'Unemployment rate and labor market statistics',
             'EUROSTAT_LFS1', 1),
            ('MPD', 'Monetary Policy Projections',
             'ECB staff macroeconomic projections',
             'ECB_MPD1', 1),
            ('GPR', 'Geopolitical Risk Indicators',
             'Country-specific geopolitical risk indices (percent of news articles)',
             'GPR_EXCEL', 18),
        ]

        with self.engine.connect() as conn:
            # Check if data already exists
            result = conn.execute(text("SELECT COUNT(*) FROM fact_table_metadata")).scalar()
            if result == 0:
                for code, name, desc, source, count in fact_tables:
                    conn.execute(text(
                        "INSERT INTO fact_table_metadata (table_code, table_name, description, dataset_source, series_count, created_at) "
                        "VALUES (:code, :name, :desc, :source, :count, :created)"
                    ), {
                        'code': code, 'name': name, 'desc': desc,
                        'source': source, 'count': count, 'created': datetime.utcnow()
                    })
                conn.commit()
                logger.info("Fact table metadata initialized")

    def _initialize_dimension_data(self):
        """Populate dimension tables with common values."""
        with self.engine.connect() as conn:
            # Frequencies
            freqs = [
                ('D', 'Daily', 'Daily observations'),
                ('B', 'Business Daily', 'Business days only'),
                ('M', 'Monthly', 'Monthly observations'),
                ('Q', 'Quarterly', 'Quarterly observations'),
                ('A', 'Annual', 'Annual observations'),
            ]
            result = conn.execute(text("SELECT COUNT(*) FROM dim_frequency")).scalar()
            if result == 0:
                for code, name, desc in freqs:
                    conn.execute(text(
                        "INSERT INTO dim_frequency (freq_code, freq_name, description) VALUES (:c, :n, :d)"
                    ), {'c': code, 'n': name, 'd': desc})
                conn.commit()
                logger.info("Dimension tables initialized")

    def parse_xml_file(self, file_path: str) -> Tuple[str, str, List[Dict]]:
        """
        Parse SDMX 2.1 XML file and extract structured data.

        Returns:
            Tuple of (dataset_code, fact_table, series_list)
        """
        logger.info(f"Parsing XML file: {file_path}")

        tree = etree.parse(file_path)
        root = tree.getroot()

        # Get dataset code
        dataset = root.find('.//message:DataSet', NAMESPACES)
        dataset_code = dataset.get('structureRef', 'UNKNOWN')

        # Determine fact table from file name and dataset
        fact_table = self._determine_fact_table(file_path, dataset_code)

        series_list = []

        # Parse each series
        for series in root.findall('.//generic:Series', NAMESPACES):
            series_data = {
                'dimensions': {},
                'attributes': {},
                'observations': []
            }

            # Extract dimensions
            for value in series.findall('.//generic:SeriesKey/generic:Value', NAMESPACES):
                series_data['dimensions'][value.get('id')] = value.get('value')

            # Extract series-level attributes
            for attr in series.findall('.//generic:Series/generic:Attributes/generic:Value', NAMESPACES):
                series_data['attributes'][attr.get('id')] = attr.get('value')

            # Extract observations
            for obs in series.findall('.//generic:Obs', NAMESPACES):
                obs_dict = {}

                # Time dimension
                obs_dim = obs.find('.//generic:ObsDimension', NAMESPACES)
                obs_dict['time_period'] = obs_dim.get('value')

                # Observation value
                obs_value = obs.find('.//generic:ObsValue', NAMESPACES)
                value = obs_value.get('value')
                try:
                    obs_dict['obs_value'] = float(value) if value != 'NaN' else None
                except (ValueError, TypeError):
                    obs_dict['obs_value'] = None

                # Observation attributes
                for attr in obs.findall('.//generic:Obs/generic:Attributes/generic:Value', NAMESPACES):
                    obs_dict[attr.get('id')] = attr.get('value')

                series_data['observations'].append(obs_dict)

            series_list.append(series_data)

        logger.info(f"Parsed {len(series_list)} series from {dataset_code} -> {fact_table}")
        return dataset_code, fact_table, series_list

    def _determine_fact_table(self, file_path: str, dataset_code: str) -> str:
        """Determine which fact table to use based on file and dataset."""
        file_name = Path(file_path).name.upper()

        if dataset_code == 'ECB_FMD2':
            if 'CISS' in file_name:
                return 'fsi'
            elif 'YC' in file_name:
                return 'yc'
            elif 'FM' in file_name:
                return 'mmr'
        elif dataset_code == 'ECB_ICP3':
            return 'inf'
        elif dataset_code == 'ECB_MIR1':
            return 'ir'
        elif dataset_code == 'EUROSTAT_LFS1':
            return 'lfi'
        elif dataset_code == 'ECB_MPD1':
            return 'mpd'

        return 'UNKNOWN'

    def ingest_file(self, file_path: str) -> Dict:
        """
        Ingest a single XML file into the database.

        Returns:
            Dictionary with ingestion statistics
        """
        dataset_code, fact_table, series_list = self.parse_xml_file(file_path)

        if fact_table == 'UNKNOWN':
            logger.error(f"Could not determine fact table for {file_path}")
            return {'success': False, 'error': 'Unknown fact table'}

        stats = {
            'file': Path(file_path).name,
            'dataset': dataset_code,
            'fact_table': fact_table,
            'series_ingested': 0,
            'observations_ingested': 0
        }

        with self.engine.connect() as conn:
            for series_data in series_list:
                # Create series key
                dims = series_data['dimensions']
                attrs = series_data['attributes']

                series_key = f"{dataset_code}:{':'.join([f'{k}={v}' for k, v in sorted(dims.items())])}"

                # Check if series exists
                result = conn.execute(
                    text("SELECT series_id FROM series_metadata WHERE series_key = :key"),
                    {'key': series_key}
                ).fetchone()

                if result:
                    series_id = result[0]
                else:
                    # Insert new series
                    obs_dates = [o['time_period'] for o in series_data['observations']]

                    result = conn.execute(text(
                        """INSERT INTO series_metadata
                        (series_key, dataset_code, fact_table, title, title_complete,
                         unit_code, unit_mult, decimals, frequency_code, time_format,
                         first_obs_date, last_obs_date, total_obs, created_at, updated_at)
                        VALUES (:key, :dataset, :table, :title, :title_comp,
                                :unit, :unit_mult, :decimals, :freq, :time_fmt,
                                :first_date, :last_date, :total_obs, :created, :updated)
                        RETURNING series_id"""
                    ), {
                        'key': series_key,
                        'dataset': dataset_code,
                        'table': fact_table,
                        'title': attrs.get('TITLE', '')[:500] if attrs.get('TITLE') else '',
                        'title_comp': attrs.get('TITLE_COMPL', ''),
                        'unit': attrs.get('UNIT', ''),
                        'unit_mult': int(attrs.get('UNIT_MULT', 0)) if attrs.get('UNIT_MULT') else 0,
                        'decimals': int(attrs.get('DECIMALS', 0)) if attrs.get('DECIMALS') else 0,
                        'freq': dims.get('FREQ', ''),
                        'time_fmt': attrs.get('TIME_FORMAT', ''),
                        'first_date': min(obs_dates) if obs_dates else None,
                        'last_date': max(obs_dates) if obs_dates else None,
                        'total_obs': len(series_data['observations']),
                        'created': datetime.utcnow(),
                        'updated': datetime.utcnow()
                    })
                    series_id = result.fetchone()[0]
                    stats['series_ingested'] += 1

                # Insert observations
                for obs in series_data['observations']:
                    conn.execute(text(f"""
                        INSERT INTO {fact_table}
                        (series_id, obs_date, obs_value, obs_status, obs_conf, created_at)
                        VALUES (:sid, :date, :value, :status, :conf, :created)
                    """), {
                        'sid': series_id,
                        'date': obs['time_period'],
                        'value': obs['obs_value'],
                        'status': obs.get('OBS_STATUS', ''),
                        'conf': obs.get('OBS_CONF', ''),
                        'created': datetime.utcnow()
                    })
                    stats['observations_ingested'] += 1

            conn.commit()

        logger.info(f"Ingested {stats['series_ingested']} series, {stats['observations_ingested']} observations into {fact_table}")
        return stats

    def ingest_directory(self, directory_path: str = "data") -> List[Dict]:
        """Ingest all XML files from a directory."""
        xml_files = list(Path(directory_path).glob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML files in {directory_path}")

        results = []
        for xml_file in xml_files:
            try:
                stats = self.ingest_file(str(xml_file))
                stats['success'] = True
                results.append(stats)
            except Exception as e:
                logger.error(f"Error processing {xml_file}: {e}")
                results.append({
                    'file': xml_file.name,
                    'success': False,
                    'error': str(e)
                })

        return results

    def load_fact_table(self, table_name: str, series_id: Optional[int] = None) -> pd.DataFrame:
        """Load data from a fact table into pandas DataFrame."""
        query = f"SELECT * FROM {table_name}"

        if series_id:
            query += f" WHERE series_id = {series_id}"

        query += " ORDER BY obs_date"

        df = pd.read_sql(text(query), self.engine)
        logger.info(f"Loaded {len(df)} observations from {table_name}")
        return df

    def get_series_info(self) -> pd.DataFrame:
        """Get information about all series."""
        query = """
        SELECT s.series_id, s.fact_table, s.title, s.unit_code,
               s.frequency_code, s.total_obs, s.first_obs_date, s.last_obs_date,
               f.table_name as fact_table_name
        FROM series_metadata s
        LEFT JOIN fact_table_metadata f ON s.fact_table = f.table_code
        ORDER BY s.fact_table, s.series_id
        """
        return pd.read_sql(text(query), self.engine)


class GPRDataIngestion:
    """Handles ingestion of GPR Excel data into PostgreSQL."""

    # Mapping of Excel column names to series IDs and country info
    COUNTRY_MAPPING = {
        'GPRC_BEL': {'series_id': 25, 'code': 'BEL', 'name': 'Belgium'},
        'GPRC_CHE': {'series_id': 26, 'code': 'CHE', 'name': 'Switzerland'},
        'GPRC_DEU': {'series_id': 27, 'code': 'DEU', 'name': 'Germany'},
        'GPRC_DNK': {'series_id': 28, 'code': 'DNK', 'name': 'Denmark'},
        'GPRC_ESP': {'series_id': 29, 'code': 'ESP', 'name': 'Spain'},
        'GPRC_FIN': {'series_id': 30, 'code': 'FIN', 'name': 'Finland'},
        'GPRC_FRA': {'series_id': 31, 'code': 'FRA', 'name': 'France'},
        'GPRC_GBR': {'series_id': 32, 'code': 'GBR', 'name': 'United Kingdom'},
        'GPRC_HUN': {'series_id': 33, 'code': 'HUN', 'name': 'Hungary'},
        'GPRC_ITA': {'series_id': 34, 'code': 'ITA', 'name': 'Italy'},
        'GPRC_NLD': {'series_id': 35, 'code': 'NLD', 'name': 'Netherlands'},
        'GPRC_NOR': {'series_id': 36, 'code': 'NOR', 'name': 'Norway'},
        'GPRC_POL': {'series_id': 37, 'code': 'POL', 'name': 'Poland'},
        'GPRC_PRT': {'series_id': 38, 'code': 'PRT', 'name': 'Portugal'},
        'GPRC_RUS': {'series_id': 39, 'code': 'RUS', 'name': 'Russia'},
        'GPRC_SWE': {'series_id': 40, 'code': 'SWE', 'name': 'Sweden'},
        'GPRC_TUR': {'series_id': 41, 'code': 'TUR', 'name': 'Turkey'},
        'GPRC_UKR': {'series_id': 42, 'code': 'UKR', 'name': 'Ukraine'},
    }

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize GPR data ingestion handler."""
        load_dotenv()

        if connection_string is None:
            connection_string = os.getenv('DATABASE_URL')
            if connection_string is None:
                raise ValueError(
                    "No database connection string provided. "
                    "Set DATABASE_URL environment variable or pass connection_string."
                )

        self.engine = create_engine(connection_string)
        logger.info("GPR ingestion: Database connection established")

    def populate_series_metadata(self):
        """Insert 18 series metadata rows for European GPR countries."""
        logger.info("Populating series_metadata with 18 GPR series...")

        with self.engine.connect() as conn:
            for excel_col, info in self.COUNTRY_MAPPING.items():
                series_id = info['series_id']
                country_code = info['code']
                country_name = info['name']

                # Check if series already exists
                result = conn.execute(
                    text("SELECT series_id FROM series_metadata WHERE series_id = :sid"),
                    {'sid': series_id}
                ).fetchone()

                if result:
                    logger.info(f"Series {series_id} (GPR_{country_code}) already exists, skipping")
                    continue

                # Insert new series
                conn.execute(text("""
                    INSERT INTO series_metadata
                    (series_id, series_key, dataset_code, fact_table, series_label,
                     unit_code, frequency_code, time_format,
                     first_obs_date, last_obs_date, total_obs,
                     created_at, updated_at)
                    VALUES (:sid, :key, :dataset, :table, :label,
                            :unit, :freq, :time_fmt,
                            :first_date, :last_date, :total_obs,
                            :created, :updated)
                """), {
                    'sid': series_id,
                    'key': f'GPR:COUNTRY={country_code}',
                    'dataset': 'GPR_DATA',
                    'table': 'gpr',
                    'label': f'GPR_{country_code}',
                    'unit': 'PC',  # Percentage
                    'freq': 'M',   # Monthly
                    'time_fmt': 'P1M',
                    'first_date': '1985-01-31',  # Month-end format
                    'last_date': '2026-02-28',
                    'total_obs': 494,
                    'created': datetime.utcnow(),
                    'updated': datetime.utcnow()
                })
                logger.info(f"Inserted series {series_id}: GPR_{country_code} - {country_name}")

            conn.commit()
        logger.info("✓ Series metadata populated successfully")

    def load_excel(self, file_path: str = 'data/data_gpr_export.xls') -> pd.DataFrame:
        """
        Load GPR data from Excel file and extract European GPRC columns.

        Args:
            file_path: Path to data_gpr_export.xls

        Returns:
            DataFrame with columns: month + 18 GPRC columns
        """
        logger.info(f"Loading GPR data from {file_path}...")

        # Read Excel file
        df = pd.read_excel(file_path)

        # Select month + 18 European GPRC columns
        european_cols = list(self.COUNTRY_MAPPING.keys())
        selected_cols = ['month'] + european_cols

        df_european = df[selected_cols].copy()

        # Filter to valid data (1985 onwards)
        df_european = df_european.dropna(subset=european_cols, how='all')

        logger.info(f"✓ Loaded {len(df_european)} observations for {len(european_cols)} countries")
        logger.info(f"  Date range: {df_european['month'].min()} to {df_european['month'].max()}")

        return df_european

    def transform_to_long_format(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        """
        Transform wide format DataFrame to long format for database insertion.

        Args:
            df_wide: DataFrame with columns [month, GPRC_BEL, GPRC_CHE, ...]

        Returns:
            DataFrame with columns [series_id, obs_date, obs_value]
        """
        logger.info("Transforming data to long format...")

        # Melt to long format
        df_long = df_wide.melt(
            id_vars=['month'],
            value_vars=list(self.COUNTRY_MAPPING.keys()),
            var_name='country_column',
            value_name='obs_value'
        )

        # Map country columns to series_id
        df_long['series_id'] = df_long['country_column'].map(
            lambda x: self.COUNTRY_MAPPING[x]['series_id']
        )

        # Convert month to month-END string format (YYYY-MM-DD)
        # Excel has month-start dates, convert to month-end for consistency with transform.py
        df_long['obs_date'] = pd.to_datetime(df_long['month']) + pd.offsets.MonthEnd(0)
        df_long['obs_date'] = df_long['obs_date'].dt.strftime('%Y-%m-%d')

        # Select and order columns
        df_long = df_long[['series_id', 'obs_date', 'obs_value']].copy()

        # Remove any NaN values
        df_long = df_long.dropna(subset=['obs_value'])

        logger.info(f"✓ Transformed to {len(df_long)} observations in long format")

        return df_long

    def insert_observations(self, df_long: pd.DataFrame):
        """
        Bulk insert observations into gpr fact table.

        Args:
            df_long: DataFrame with columns [series_id, obs_date, obs_value]
        """
        logger.info(f"Inserting {len(df_long)} observations into gpr table...")

        # Add metadata columns
        df_long['obs_status'] = ''
        df_long['obs_conf'] = ''
        df_long['created_at'] = datetime.utcnow()

        # Bulk insert using pandas to_sql
        df_long.to_sql(
            name='gpr',
            con=self.engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )

        logger.info(f"✓ Successfully inserted {len(df_long)} observations")

    def run_full_ingestion(self, file_path: str = 'data/data_gpr_export.xls'):
        """
        Orchestrate full GPR data ingestion pipeline.

        Steps:
        1. Populate series_metadata with 18 European countries
        2. Load Excel data
        3. Transform to long format
        4. Insert observations into gpr table

        Args:
            file_path: Path to data_gpr_export.xls
        """
        logger.info("=" * 80)
        logger.info("GPR DATA INGESTION PIPELINE")
        logger.info("=" * 80)

        # Step 1: Populate series metadata
        self.populate_series_metadata()

        # Step 2: Load Excel data
        df_wide = self.load_excel(file_path)

        # Step 3: Transform to long format
        df_long = self.transform_to_long_format(df_wide)

        # Step 4: Insert observations
        self.insert_observations(df_long)

        # Summary
        logger.info("=" * 80)
        logger.info("GPR INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ 18 European country series added (series_id 25-42)")
        logger.info(f"✓ {len(df_long)} observations inserted into gpr table")
        logger.info(f"✓ Date range: {df_long['obs_date'].min()} to {df_long['obs_date'].max()}")
        logger.info("=" * 80)

        return {
            'series_count': 18,
            'observations_count': len(df_long),
            'first_date': df_long['obs_date'].min(),
            'last_date': df_long['obs_date'].max()
        }


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description='ECB Data Ingestion Pipeline')
    parser.add_argument('--create-schema', action='store_true', help='Create database schema')
    parser.add_argument('--ingest', help='Ingest XML files from directory')
    parser.add_argument('--ingest-gpr', help='Ingest GPR data from Excel file (default: data/data_gpr_export.xls)')
    parser.add_argument('--list-series', action='store_true', help='List all series')
    parser.add_argument('--list-tables', action='store_true', help='List fact tables')

    args = parser.parse_args()

    ingestion = ECBDataIngestion()

    if args.create_schema:
        logger.info("Creating database schema...")
        ingestion.create_schema()
        logger.info("Schema created successfully")

    if args.ingest_gpr:
        # Handle GPR data ingestion
        gpr_file = args.ingest_gpr if args.ingest_gpr != 'True' else 'data/data_gpr_export.xls'
        logger.info(f"Starting GPR data ingestion from {gpr_file}...")
        gpr_ingestion = GPRDataIngestion()
        result = gpr_ingestion.run_full_ingestion(gpr_file)

        print("\n" + "="*80)
        print("GPR INGESTION SUMMARY")
        print("="*80)
        print(f"✓ Series added: {result['series_count']} European countries (series_id 25-42)")
        print(f"✓ Observations: {result['observations_count']}")
        print(f"✓ Date range: {result['first_date']} to {result['last_date']}")
        print("="*80)

    if args.ingest:
        logger.info(f"Ingesting files from {args.ingest}...")
        results = ingestion.ingest_directory(args.ingest)

        print("\n" + "="*80)
        print("INGESTION SUMMARY")
        print("="*80)
        for r in results:
            if r['success']:
                print(f"✓ {r['file']:40s} -> {r['fact_table']:5s} "
                      f"({r['series_ingested']} series, {r['observations_ingested']} obs)")
            else:
                print(f"✗ {r['file']:40s} ERROR: {r.get('error', 'Unknown')}")

    if args.list_series:
        df = ingestion.get_series_info()
        print("\n" + "="*80)
        print(f"SERIES INVENTORY ({len(df)} series)")
        print("="*80)
        print(df.to_string(index=False))

    if args.list_tables:
        query = "SELECT * FROM fact_table_metadata ORDER BY table_code"
        df = pd.read_sql(text(query), ingestion.engine)
        print("\n" + "="*80)
        print("FACT TABLES")
        print("="*80)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()