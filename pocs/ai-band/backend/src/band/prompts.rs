pub fn drums_prompt(round: usize, genre: &str, context: &str) -> String {
    format!(
        "You are a world-class drummer playing {genre} music. Round {round} of a jam session.\n\
        Previous parts:\n{context}\n\n\
        Write a REAL {genre} drum groove in ABC notation. This MUST sound like authentic {genre} drumming.\n\
        Use syncopation, ghost notes, and rhythmic variation. NO robotic straight eighth notes.\n\
        Add fills at the end of every 4 bars. Vary dynamics with !p! !f! !mf! markings.\n\
        Use swing feel for jazz/funk, driving power for rock/metal, laid back for reggae/soul.\n\
        Use %%MIDI channel 10 for drums.\n\
        Keep it 8-16 bars. Use z for rests, vary note lengths (eighth, sixteenth, triplets).\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        The X: field MUST be X:1\n\
        Make it GROOVE hard in the style of {genre}.",
        genre = genre, round = round, context = context
    )
}

pub fn bass_prompt(round: usize, genre: &str, context: &str) -> String {
    format!(
        "You are a world-class bass player playing {genre} music. Round {round} of a jam session.\n\
        Previous parts:\n{context}\n\n\
        Write a REAL {genre} bass line in ABC notation.\n\
        Use %%MIDI channel 1 and %%MIDI program 33 for electric bass.\n\
        This MUST sound like authentic {genre} bass playing.\n\
        Use slides, hammer-ons (grace notes), syncopated rhythms, and walking patterns.\n\
        Mix eighth notes, sixteenth notes, and rests for groove. NO boring whole notes or half notes only.\n\
        For rock: use power note patterns and root-fifth movement.\n\
        For funk: use slap-style sixteenth note patterns with ghost notes.\n\
        For jazz: use walking bass lines.\n\
        Use bass clef, keep it 8-16 bars. Use low octave notes (C, D, E, etc).\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        The X: field MUST be X:1\n\
        Lock in tight with the drums and make it {genre} to the bone.",
        genre = genre, round = round, context = context
    )
}

pub fn guitar_prompt(round: usize, genre: &str, context: &str) -> String {
    format!(
        "You are a world-class guitar player playing {genre} music. Round {round} of a jam session.\n\
        Previous parts:\n{context}\n\n\
        Write a {genre} guitar part in ABC notation.\n\
        Use %%MIDI channel 1 and %%MIDI program 30 for distortion guitar (rock/metal) or program 26 for clean guitar (jazz/funk).\n\
        This MUST sound like authentic {genre} guitar playing.\n\
        For rock/metal: use power chords, palm muting rhythms, and riffs with attitude.\n\
        For funk: use choppy sixteenth note chord stabs and muted strums.\n\
        For jazz: use extended chord voicings and comping patterns.\n\
        For blues: use pentatonic licks and bends.\n\
        Use syncopation, varied rhythms, rests between phrases. NO straight quarter note strumming.\n\
        Add dynamics with !f! !mf! !p! markings.\n\
        Use treble clef, keep it 8-16 bars.\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        The X: field MUST be X:1\n\
        Make it rip in the style of {genre}.",
        genre = genre, round = round, context = context
    )
}

pub fn melody_prompt(round: usize, genre: &str, context: &str) -> String {
    format!(
        "You are a world-class melody writer composing {genre} music. Round {round} of a jam session.\n\
        Previous parts:\n{context}\n\n\
        Write a {genre} melody/lead line in ABC notation.\n\
        Use %%MIDI channel 1 and %%MIDI program 1 for piano melody.\n\
        This MUST sound like authentic {genre} melody.\n\
        Use expressive phrasing: mix long sustained notes with fast runs.\n\
        Add grace notes, ornaments, and rhythmic variation.\n\
        Use call-and-response patterns. Leave space (rests) for the melody to breathe.\n\
        For rock: use pentatonic hooks and power riffs.\n\
        For jazz: use chromatic passing tones and bebop lines.\n\
        For pop: use memorable, singable hooks with repetition.\n\
        Use treble clef, keep it 8-16 bars. Make it catchy and singable.\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        The X: field MUST be X:1\n\
        Complement the drums, bass, and guitar. Make it unmistakably {genre}.",
        genre = genre, round = round, context = context
    )
}

pub fn lyrics_prompt(round: usize, genre: &str, context: &str, lyrics_theme: &str) -> String {
    let theme_line = if lyrics_theme.is_empty() {
        String::new()
    } else {
        format!("\nIMPORTANT: The lyrics MUST be about this theme: {}\n", lyrics_theme)
    };
    format!(
        "You are a world-class {genre} songwriter. Round {round} of collaboration.\n\
        The music so far:\n{context}\n{theme_line}\n\
        Write lyrics that fit the melody and feel authentically {genre}.\n\
        Use ABC notation 'w:' lines for words.\n\
        Match the syllable count to the melody rhythm. Use natural speech patterns.\n\
        For rock: raw, energetic, rebellious lyrics.\n\
        For jazz: poetic, sophisticated, smooth lyrics.\n\
        For funk: groove-oriented, playful, rhythmic lyrics.\n\
        For pop: catchy, relatable, hook-driven lyrics.\n\
        Output ONLY the lyrics as w: lines that can be appended to the melody ABC notation.\n\
        Keep it 4-8 lines. Make it creative and authentically {genre}.",
        genre = genre, round = round, context = context, theme_line = theme_line
    )
}

pub fn singer_prompt(genre: &str, context: &str) -> String {
    format!(
        "You are a singer performing the final version of a {genre} song.\n\
        Here is the full song with all parts:\n{context}\n\n\
        Write a vocal part in ABC notation that a {genre} singer would perform.\n\
        Use %%MIDI channel 1 and %%MIDI program 54 for voice.\n\
        Include the w: lyric lines under the vocal melody notes.\n\
        The vocal melody should follow the main melody but simplified for singing.\n\
        Add phrasing, dynamics (!f! !p! !mf!), and expression marks.\n\
        Use treble clef. The X: field MUST be X:1\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        Make the vocal delivery match {genre} style.",
        genre = genre, context = context
    )
}
