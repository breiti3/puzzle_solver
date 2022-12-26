use opencv::{
    Result,
    prelude::*,
    improc::*,
    imgcodecs::*,
    core,
    highgui
};

use std::path::Path;



fn main() -> Result<()>{
    let path: PathBuf = ["..", "..", "puzzle_bilder","test2.jpg"].iter().collect();
    let im = imgcodecs::imread(path);
    highgui::named_window("hello opencv!", 0).unwrap();
    highgui::imshow("hello opencv!", &im).unwrap();
    highgui::wait_key(10000).unwrap();
    Ok(())
}
