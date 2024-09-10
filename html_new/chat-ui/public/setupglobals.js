const urlParams = new URLSearchParams(window.location.search);
window.google = urlParams.get('google');
window.dropbox = urlParams.get('dropbox');
window.fullname = urlParams.get('fullname');
window.picture = urlParams.get('picture');
if (window.picture == null) {
    window.picture = "./user.png";
}
console.log("index.html: google=" + window.google
                    + ", dropbox=" + window.dropbox
                    + ", fullname=" + window.fullname
                    + ", picture=" + window.picture)
