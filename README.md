# Kinetics-data-processing
Quick and dirty implementation. Assumed that given video scale can always be scaled to 360p. Used async/aiohttp for just the url fetching and tested with 1 endpoint(1000 videos). 
Used Ray to organize the actual transcoding and writing to lance since those are more CPU-bound tasks rather then the URL. Attempted to the depth extraction but it was wayyy too heavy for my poor laptop but I left the commented code there anyway. 

In production, there are a huge amount of considerations, questions, changes to think about. Depending on how important data quality is, you would have to block on failures transcoding to prevent loss frames. There is a way to stream/pipe into ffpmeg but I couldn't figure it out but if successfull that would increase the throughout and decrease in memory usage. The biggest bounds here are RAM, everything else can be distributed using Ray but putting models into Ray questionable consideration as well unless the server/machine is high CPU/GPU otherwise the process itself will get slowed down

Overall very fun! Definitely took me more then an hour 
