#!/bin/bash
set -e

if [ x"$1" == "x" ] ; then
    echo Usage: $0 dir_to_copy_files_to
    exit 255
fi

DESTDIR="$1"

/bin/rm -rf "${DESTDIR}"
/bin/mkdir -p "${DESTDIR}"/html/errors

/bin/rm -f /tmp/serviceconfig.js
echo "var ServiceConfig = {" > /tmp/serviceconfig.js
echo "  INDEX_DIR: \"/yoja-index-dir\"," >> /tmp/serviceconfig.js
echo "  DOCS_DIR: \"/yoja-docs-dir\"," >> /tmp/serviceconfig.js
echo "  OAUTH_REDIRECT_URI: \"notused_oauth_redirect_uri\"," >> /tmp/serviceconfig.js
echo "  envAPIEndpoint: \"rest\"," >> /tmp/serviceconfig.js
echo "  envAPIKey: \"unused\"," >> /tmp/serviceconfig.js
echo "  GOOGLE_CLIENT_ID: \"notused_google_client_id\"," >> /tmp/serviceconfig.js
echo "  DROPBOX_CLIENT_ID: \"notused_dropbox_client_id\"" >> /tmp/serviceconfig.js
echo "}" >> /tmp/serviceconfig.js

/bin/cp -f /tmp/serviceconfig.js "${DESTDIR}"/html/serviceconfig.js
/bin/rm -f /tmp/serviceconfig.js

FILES="avatar.jpg icon.png login.html login.js login.css Dropbox-sdk.min.js"
for i in $FILES
do
    echo Copying $i
    /bin/cp -f $i "${DESTDIR}"/html/
done

EFILES="404.html 405.html"
for i in $EFILES
do
    echo Copying $i
    /bin/cp -f $i "${DESTDIR}"/html/errors/
done

(cd chat-ui; rm -rf build)
(cd chat-ui; rm -rf node_modules)
(cd chat-ui; npm install --force)

(cd chat-ui/; npm run build)
(cd chat-ui/build; rsync -avz . ${DESTDIR}/html/)

exit 0
