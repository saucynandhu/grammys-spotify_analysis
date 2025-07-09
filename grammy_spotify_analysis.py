import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
from collections import defaultdict
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set the style for the plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Load the datasets
def load_datasets():
    """Load all the required datasets."""
    logger.info("Loading datasets...")
    
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(current_dir, 'datasets')
        
        # Load Grammy data
        grammy_path = os.path.join(datasets_dir, 'Grammy Award Nominees and Winners 1958-2024.csv')
        logger.info(f"Loading Grammy data from: {grammy_path}")
        grammy = pd.read_csv(grammy_path, encoding='latin1')
        logger.info(f"Grammy data loaded. Shape: {grammy.shape}")
        
        # Load Spotify most streamed songs
        spotify_path = os.path.join(datasets_dir, 'Spotify most streamed.csv')
        logger.info(f"Loading Spotify data from: {spotify_path}")
        spotify_songs = pd.read_csv(spotify_path, 
                                  thousands=',',
                                  encoding='latin1')
        logger.info(f"Spotify data loaded. Shape: {spotify_songs.shape}")
        
        # Load artists data
        artists_path = os.path.join(datasets_dir, 'artists.csv')
        logger.info(f"Loading artists data from: {artists_path}")
        artists = pd.read_csv(artists_path, 
                             thousands=',',
                             encoding='latin1')
        logger.info(f"Artists data loaded. Shape: {artists.shape}")
        
        # Load producer data
        producers_path = os.path.join(datasets_dir, 'Supplementary Table Producer of the Year 2019-2024.csv')
        logger.info(f"Loading producers data from: {producers_path}")
        producers = pd.read_csv(producers_path, encoding='latin1')
        logger.info(f"Producers data loaded. Shape: {producers.shape}")
        
        logger.info("All datasets loaded successfully!")
        return grammy, spotify_songs, artists, producers
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def clean_artist_name(artist):
    """Clean artist names by removing featured artists and extra information."""
    if pd.isna(artist):
        return ""
    # Remove content in parentheses and brackets
    artist = re.sub(r'\([^)]*\)', '', artist)
    artist = re.sub(r'\[[^\]]*\]', '', artist)
    # Remove 'Featuring' and everything after it
    artist = re.sub(r'Feat.*$', '', artist, flags=re.IGNORECASE)
    artist = re.sub(r'Ft\.?.*$', '', artist, flags=re.IGNORECASE)
    # Remove common suffixes and clean up
    artist = re.sub(r'&.*$', '', artist)  # Remove '& The Weeknd' etc.
    artist = re.sub(r',.*$', '', artist)  # Remove anything after comma
    artist = re.sub(r'\s+', ' ', artist).strip()  # Clean up whitespace
    return artist.strip()

def analyze_grammy_winners(grammy):
    """Analyze Grammy winners data."""
    print("\n=== Grammy Winners Analysis ===")
    
    # Filter only winners
    winners = grammy[grammy['Winner'] == True].copy()
    
    # 1. Most awarded artists
    top_artists = winners['Nominee'].value_counts().head(10)
    print("\nTop 10 Most Awarded Artists:")
    print(top_artists)
    
    # 2. Awards by year
    awards_by_year = winners['Year'].value_counts().sort_index()
    
    # Plot awards by year
    plt.figure(figsize=(12, 6))
    awards_by_year.plot(kind='line', marker='o')
    plt.title('Number of Grammy Awards by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Awards')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('awards_by_year.png')
    plt.close()
    
    return winners

def analyze_spotify_data(spotify_songs):
    """Analyze Spotify streaming data."""
    print("\n=== Spotify Streaming Analysis ===")
    
    # Clean artist names (extract main artist)
    spotify_songs['Main Artist'] = spotify_songs['Artist and Title'].apply(
        lambda x: x.split(' - ')[0] if ' - ' in x else x.split('–')[0].strip()
    )
    
    # 1. Most streamed artists
    artist_streams = spotify_songs.groupby('Main Artist')['Streams'].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 Most Streamed Artists on Spotify:")
    print(artist_streams)
    
    # 2. Top songs by daily streams
    top_daily = spotify_songs.sort_values('Daily', ascending=False).head(10)
    print("\nTop 10 Songs by Daily Streams:")
    print(top_daily[['Artist and Title', 'Daily']])
    
    return spotify_songs

def cross_analysis(grammy_winners, spotify_data, artists_data):
    """Perform cross-dataset analysis between Grammy winners and streaming data."""
    print("\n=== Cross-Dataset Analysis ===")
    
    # Clean artist names in both datasets for better matching
    grammy_winners['Clean Nominee'] = grammy_winners['Nominee'].apply(clean_artist_name)
    spotify_data['Clean Artist'] = spotify_data['Main Artist'].apply(clean_artist_name)
    
    # Find Grammy winners in Spotify data
    grammy_artists = set(grammy_winners['Clean Nominee'].str.lower())
    spotify_data['Is_Grammy_Winner'] = spotify_data['Clean Artist'].str.lower().isin(grammy_artists)
    
    # Compare streams for Grammy winners vs non-winners
    winner_stats = spotify_data.groupby('Is_Grammy_Winner')['Streams'].describe()
    print("\nStreaming Statistics: Grammy Winners vs Non-Winners")
    print(winner_stats)
    
    # Get top Grammy winners by streams
    top_grammy_streamed = spotify_data[spotify_data['Is_Grammy_Winner']].sort_values(
        'Streams', ascending=False).head(10)
    
    print("\nTop 10 Grammy Winners by Spotify Streams:")
    print(top_grammy_streamed[['Artist and Title', 'Streams']])
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Is_Grammy_Winner', y='Streams', data=spotify_data.head(500))  # First 500 for visualization
    plt.yscale('log')
    plt.title('Streaming Distribution: Grammy Winners vs Non-Winners')
    plt.xticks([0, 1], ['Non-Winners', 'Winners'])
    plt.tight_layout()
    plt.savefig('grammy_vs_streams.png')
    plt.close()

def analyze_producers(producers_data, spotify_data):
    """Analyze producer data and its relation to streaming success."""
    print("\n=== Producer Analysis ===")
    
    # Focus on non-classical producers
    prod = producers_data[producers_data['Award Name'] == 'Producer Of The Year, Non-Classical'].copy()
    
    # Extract all works and their artists
    works = []
    for _, row in prod.iterrows():
        work_entries = [w.strip() for w in row['Work'].split('•') if w.strip()]
        for work in work_entries:
            # Extract song/album and artist if available
            if '(' in work and ')' in work.split('(')[-1]:
                work_name = work.split('(')[0].strip()
                artist = work.split('(')[-1].split(')')[0].strip()
                works.append({
                    'Producer': row['Nominee'],
                    'Work': work_name,
                    'Artist': artist,
                    'Year': row['Year'],
                    'Winner': row['Winner']
                })
    
    works_df = pd.DataFrame(works)
    
    # Find these works in Spotify data
    works_df['In_Spotify_Top'] = works_df.apply(
        lambda x: x['Work'].lower() in ' '.join(spotify_data['Artist and Title'].str.lower()), 
        axis=1
    )
    
    # Analyze success rate of producers
    producer_success = works_df.groupby('Producer').agg({
        'Winner': 'sum',
        'In_Spotify_Top': 'sum',
        'Work': 'count'
    }).rename(columns={'Work': 'Total_Productions'})
    
    producer_success['Win_Rate'] = producer_success['Winner'] / producer_success['Total_Productions']
    producer_success['Spotify_Hit_Rate'] = producer_success['In_Spotify_Top'] / producer_success['Total_Productions']
    
    print("\nProducer Success Analysis:")
    print(producer_success.sort_values('Win_Rate', ascending=False).head(10))

def main():
    try:
        logger.info("Starting analysis...")
        
        # Load all datasets
        grammy, spotify_songs, artists, producers = load_datasets()
        
        # Perform individual analyses
        logger.info("Analyzing Grammy winners...")
        grammy_winners = analyze_grammy_winners(grammy)
        
        logger.info("Analyzing Spotify data...")
        spotify_data = analyze_spotify_data(spotify_songs)
        
        # Perform cross-dataset analysis
        logger.info("Performing cross-dataset analysis...")
        cross_analysis(grammy_winners, spotify_data, artists)
        
        # Analyze producers
        logger.info("Analyzing producers...")
        analyze_producers(producers, spotify_data)
        
        logger.info("\nAnalysis complete! Check the generated plots for visualizations.")
        
    except Exception as e:
        logger.error(f"An error occurred during analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
