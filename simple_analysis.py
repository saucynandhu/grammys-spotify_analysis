import pandas as pd
import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_datasets():
    """Load and verify the datasets."""
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(current_dir, 'datasets')
        
        # Check if datasets directory exists
        if not os.path.exists(datasets_dir):
            logger.error(f"Datasets directory not found: {datasets_dir}")
            return None, None, None, None
        
        # List all files in the datasets directory
        logger.info("Files in datasets directory:")
        for f in os.listdir(datasets_dir):
            logger.info(f"- {f}")
        
        # Try to load Grammy data
        grammy_path = os.path.join(datasets_dir, 'Grammy Award Nominees and Winners 1958-2024.csv')
        if os.path.exists(grammy_path):
            logger.info(f"Loading Grammy data from: {grammy_path}")
            grammy = pd.read_csv(grammy_path, encoding='latin1')
            logger.info(f"Grammy data loaded. Shape: {grammy.shape}")
        else:
            logger.error(f"Grammy data file not found: {grammy_path}")
            grammy = None
        
        # Try to load Spotify data
        spotify_path = os.path.join(datasets_dir, 'Spotify most streamed.csv')
        if os.path.exists(spotify_path):
            logger.info(f"Loading Spotify data from: {spotify_path}")
            spotify = pd.read_csv(spotify_path, thousands=',', encoding='latin1')
            logger.info(f"Spotify data loaded. Shape: {spotify.shape}")
        else:
            logger.error(f"Spotify data file not found: {spotify_path}")
            spotify = None
        
        # Try to load artists data
        artists_path = os.path.join(datasets_dir, 'artists.csv')
        if os.path.exists(artists_path):
            logger.info(f"Loading artists data from: {artists_path}")
            artists = pd.read_csv(artists_path, thousands=',', encoding='latin1')
            logger.info(f"Artists data loaded. Shape: {artists.shape}")
        else:
            logger.error(f"Artists data file not found: {artists_path}")
            artists = None
        
        # Try to load producers data
        producers_path = os.path.join(datasets_dir, 'Supplementary Table Producer of the Year 2019-2024.csv')
        if os.path.exists(producers_path):
            logger.info(f"Loading producers data from: {producers_path}")
            producers = pd.read_csv(producers_path, encoding='latin1')
            logger.info(f"Producers data loaded. Shape: {producers.shape}")
        else:
            logger.error(f"Producers data file not found: {producers_path}")
            producers = None
        
        return grammy, spotify, artists, producers
        
    except Exception as e:
        logger.error(f"Error in load_datasets: {str(e)}")
        return None, None, None, None

def main():
    logger.info("Starting analysis...")
    
    # Load datasets
    grammy, spotify, artists, producers = load_datasets()
    
    # Basic analysis
    if grammy is not None:
        logger.info("\n=== Basic Grammy Data Analysis ===")
        logger.info(f"Total Grammy records: {len(grammy)}")
        logger.info(f"Columns: {', '.join(grammy.columns)}")
        logger.info("\nFirst 5 rows of Grammy data:")
        logger.info(grammy.head().to_string())
    
    if spotify is not None:
        logger.info("\n=== Basic Spotify Data Analysis ===")
        logger.info(f"Total Spotify records: {len(spotify)}")
        logger.info(f"Columns: {', '.join(spotify.columns)}")
        logger.info("\nTop 5 most streamed songs:")
        logger.info(spotify.head().to_string())
    
    if artists is not None:
        logger.info("\n=== Basic Artists Data Analysis ===")
        logger.info(f"Total artist records: {len(artists)}")
        logger.info(f"Columns: {', '.join(artists.columns)}")
        logger.info("\nTop 5 artists by streams:")
        logger.info(artists.sort_values('Streams', ascending=False).head().to_string())
    
    if producers is not None:
        logger.info("\n=== Basic Producers Data Analysis ===")
        logger.info(f"Total producer records: {len(producers)}")
        logger.info(f"Columns: {', '.join(producers.columns)}")
        logger.info("\nFirst 5 rows of producers data:")
        logger.info(producers.head().to_string())
    
    logger.info("\nAnalysis complete!")

if __name__ == "__main__":
    main()
