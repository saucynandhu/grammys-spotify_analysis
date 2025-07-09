import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import defaultdict
from datetime import datetime

# Set up the plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Create output directory for plots
os.makedirs('analysis_plots', exist_ok=True)

def clean_artist_name(artist):
    """Clean artist names by removing featured artists and extra information."""
    if pd.isna(artist):
        return ""
    # Remove content in parentheses and brackets
    artist = re.sub(r'\([^)]*\)', '', artist)
    artist = re.sub(r'\[[^]]*\]', '', artist)
    # Remove common suffixes
    artist = re.sub(r'\s+feat\..*$', '', artist, flags=re.IGNORECASE)
    artist = re.sub(r'\s+&.*$', '', artist, flags=re.IGNORECASE)
    artist = re.sub(r'\s+x\s+.*$', '', artist, flags=re.IGNORECASE)
    # Clean up whitespace
    artist = artist.strip()
    return artist

def load_and_preprocess_data():
    """Load and preprocess all datasets."""
    print("Loading and preprocessing data...")
    
    # Load datasets
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    
    # Load Grammy data
    grammy = pd.read_csv(os.path.join(datasets_dir, 'Grammy Award Nominees and Winners 1958-2024.csv'), 
                        encoding='latin1')
    
    # Load Spotify most streamed songs
    spotify_songs = pd.read_csv(os.path.join(datasets_dir, 'Spotify most streamed.csv'), 
                              thousands=',',
                              encoding='latin1')
    
    # Load artists data
    artists = pd.read_csv(os.path.join(datasets_dir, 'artists.csv'), 
                         thousands=',',
                         encoding='latin1')
    
    # Load producer data
    producers = pd.read_csv(os.path.join(datasets_dir, 'Supplementary Table Producer of the Year 2019-2024.csv'),
                          encoding='latin1')
    
    # Clean artist names in all datasets
    grammy['clean_nominee'] = grammy['Nominee'].apply(clean_artist_name)
    spotify_songs['clean_artist'] = spotify_songs['Artist and Title'].apply(
        lambda x: clean_artist_name(x.split(' - ')[0]) if isinstance(x, str) else ""
    )
    artists['clean_artist'] = artists['Artist'].apply(clean_artist_name)
    
    # Extract song titles from Spotify data
    spotify_songs['song'] = spotify_songs['Artist and Title'].apply(
        lambda x: x.split(' - ')[1] if isinstance(x, str) and ' - ' in x else ""
    )
    
    # Convert streams to numeric (in millions)
    artists['streams_millions'] = artists['Streams'] / 1000
    spotify_songs['streams_millions'] = spotify_songs['Streams'] / 1000000
    
    return grammy, spotify_songs, artists, producers

def analyze_grammy_vs_streaming(grammy, spotify_songs, artists):
    """Analyze relationship between Grammy wins and streaming success."""
    print("\n=== Grammy Winners vs. Streaming Success ===")
    
    # Get unique Grammy winners
    winners = grammy[grammy['Winner'] == True]['clean_nominee'].unique()
    
    # Check which winners are in Spotify data
    spotify_winners = spotify_songs[spotify_songs['clean_artist'].isin(winners)]
    non_grammy_artists = spotify_songs[~spotify_songs['clean_artist'].isin(winners)]
    
    # Basic stats
    print(f"\nTotal Grammy winners: {len(winners)}")
    print(f"Grammy winners in Spotify top streamed: {len(spotify_winners['clean_artist'].unique())}")
    
    # Compare streaming numbers
    if not spotify_winners.empty and not non_grammy_artists.empty:
        avg_streams_winners = spotify_winners['streams_millions'].mean()
        avg_streams_non_winners = non_grammy_artists['streams_millions'].mean()
        
        print(f"\nAverage streams (millions) for Grammy winners: {avg_streams_winners:.2f}")
        print(f"Average streams (millions) for non-winners: {avg_streams_non_winners:.2f}")
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_grammy_winner', 
                   y='streams_millions',
                   data=pd.concat([
                       spotify_winners.assign(is_grammy_winner='Grammy Winner'),
                       non_grammy_artists.assign(is_grammy_winner='Non-Winner')
                   ]))
        plt.title('Streaming Numbers: Grammy Winners vs. Non-Winners')
        plt.xlabel('')
        plt.ylabel('Streams (Millions)')
        plt.tight_layout()
        plt.savefig('analysis_plots/grammy_vs_streaming.png')
        plt.close()
        
        # Find highly streamed artists without Grammys
        threshold = spotify_winners['streams_millions'].quantile(0.9)  # Top 10% of winners
        overlooked = non_grammy_artists[
            non_grammy_artists['streams_millions'] > threshold
        ].sort_values('streams_millions', ascending=False)
        
        if not overlooked.empty:
            print("\nHighly streamed artists without Grammy wins:")
            print(overlooked[['Artist and Title', 'streams_millions']].head(10))
            
            # Save to CSV for further analysis
            overlooked.to_csv('analysis_plots/overlooked_artists.csv', index=False)
    
    return spotify_winners, non_grammy_artists

def analyze_producer_impact(producers, spotify_songs):
    """Analyze the impact of award-winning producers."""
    print("\n=== Producer Impact Analysis ===")
    
    # Filter for Producer of the Year winners
    producer_winners = producers[producers['Winner'] == True].copy()
    
    if not producer_winners.empty:
        # Extract producer names (simplified - would need more robust extraction)
        producer_winners['producer_name'] = producer_winners['Nominee'].str.split('(').str[0].str.strip()
        
        # This is a simplified analysis - in reality would need to match producers to songs
        print("\nProducer of the Year winners:")
        print(producer_winners[['Year', 'producer_name']].drop_duplicates())
        
        # Would need a more sophisticated matching here
        print("\nNote: More detailed producer-artist matching would require additional data.")
    
    return producer_winners

def analyze_genre_trends(grammy, spotify_songs):
    """Analyze genre trends across Grammys and streaming."""
    print("\n=== Genre Analysis ===")
    
    # Extract genres from award names (simplified)
    grammy['award_genre'] = grammy['Award Name'].str.extract(r'(Pop|Rock|Rap|R&B|Country|Jazz|Classical|Dance|Latin|Alternative|Metal|Gospel|Reggae)', 
                                                           flags=re.IGNORECASE, expand=False)
    
    # Count awards by genre
    genre_counts = grammy['award_genre'].value_counts().head(10)
    
    # Plot top genres
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='bar')
    plt.title('Top 10 Most Common Grammy Award Genres')
    plt.xlabel('Genre')
    plt.ylabel('Number of Awards')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('analysis_plots/grammy_genres.png')
    plt.close()
    
    print("\nTop Grammy award genres:")
    print(genre_counts)
    
    # Note: Would need genre data from Spotify for a proper comparison
    print("\nNote: Genre comparison with streaming data would require genre information in the Spotify dataset.")

def analyze_award_impact(grammy, spotify_songs):
    """Analyze the impact of winning a Grammy on streaming numbers."""
    print("\n=== Grammy Award Impact on Streaming ===")
    
    # This would require time-series data of streaming numbers around award dates
    # For now, we can only do a basic correlation
    
    # Get Grammy winners and their years
    winners = grammy[grammy['Winner'] == True][['Year', 'clean_nominee']].drop_duplicates()
    
    # Match with Spotify data (simplified)
    spotify_winners = spotify_songs[spotify_songs['clean_artist'].isin(winners['clean_nominee'])]
    
    if not spotify_winners.empty:
        # Basic correlation (simplified)
        print(f"Number of Grammy winners in top streamed: {len(spotify_winners['clean_artist'].unique())}")
        
        # Would need more data for proper pre/post analysis
        print("\nNote: For detailed pre/post award analysis, we would need daily streaming data around award dates.")
    else:
        print("No Grammy winners found in the top streamed songs.")

def main():
    # Load and preprocess data
    grammy, spotify_songs, artists, producers = load_and_preprocess_data()
    
    # Perform analyses
    analyze_grammy_vs_streaming(grammy, spotify_songs, artists)
    analyze_producer_impact(producers, spotify_songs)
    analyze_genre_trends(grammy, spotify_songs)
    analyze_award_impact(grammy, spotify_songs)
    
    print("\nAnalysis complete! Check the 'analysis_plots' directory for visualizations.")

if __name__ == "__main__":
    main()
