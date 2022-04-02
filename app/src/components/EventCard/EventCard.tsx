import { Card, CardContent, Typography } from "@mui/material";
import React from "react";
import CEvent from '../../models/CustomEvent';
import './EventCard.css';

interface EventCardProps {
    event: CEvent;
}

function EventCard(props: EventCardProps) {
    return (
        <Card className="event-card">
            <CardContent>
                <Typography variant="h5" component="h2"> 
                    {props.event.event_id}
                </Typography>
                <div>
                    <Typography variant="body2" component="p">User ID: {props.event.user_id}</Typography>
                    <Typography variant="body2" component="p">Start Time: {props.event.start_time}</Typography>
                    <Typography variant="body2" component="p">Location : {props.event.city}</Typography>
                    <Typography variant="body2" component="p">Rating : {props.event.rating}</Typography>

                </div>
            </CardContent>
        </Card>
    )
};

export default EventCard